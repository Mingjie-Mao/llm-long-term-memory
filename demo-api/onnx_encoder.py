"""A drop-in replacement for `embed.Encoder` that runs on onnxruntime.

Same surface — `.dim`, `.encode(list[str]) -> np.ndarray` of L2-normalized float32 —
so `HybridRetriever` and `NumpyFlatIndex` cannot tell the difference. What it drops is
PyTorch: roughly a gigabyte of wheel that exists here only to multiply matrices.

Its output is *close to* the torch encoder's, not identical (see `export_onnx.py`,
which measures the gap and refuses to install below a threshold). Close enough to
serve a playground, and never close enough to mix into a measured store.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


class OnnxEncoder:
    def __init__(self, model_dir: str | Path, batch_size: int = 32, max_length: int = 256) -> None:
        import onnxruntime as ort
        from tokenizers import Tokenizer

        self.dir = Path(model_dir)
        self.batch_size = batch_size
        self.max_length = max_length

        self._tok = Tokenizer.from_file(str(self.dir / "tokenizer.json"))
        self._tok.enable_truncation(max_length=max_length)
        self._tok.enable_padding()

        # One thread each. The demo is latency-insensitive and memory-sensitive, and
        # onnxruntime's default pools size themselves to the host's core count, which
        # on a shared 512MB instance buys contention rather than speed.
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        self._session = ort.InferenceSession(
            str(self.dir / "model.onnx"), opts, providers=["CPUExecutionProvider"]
        )
        self._dim = int(self._session.get_outputs()[0].shape[-1])
        # Exports differ in their signature. The one produced by `export_onnx.py`
        # takes two inputs; the one HuggingFace publishes for the same model takes a
        # third, `token_type_ids`. Read the names off the session rather than assuming
        # either: feeding the wrong set fails loudly, and quietly omitting an input a
        # model declares is how an encoder starts returning subtly different vectors.
        self._inputs = {i.name for i in self._session.get_inputs()}

    @property
    def dim(self) -> int:
        return self._dim

    def encode(self, texts: list[str], show_progress: bool = False) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        out = [
            self._encode_batch(texts[i : i + self.batch_size])
            for i in range(0, len(texts), self.batch_size)
        ]
        return np.vstack(out).astype(np.float32)

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]

    def _encode_batch(self, texts: list[str]) -> np.ndarray:
        encoded = self._tok.encode_batch(texts)
        ids = np.array([e.ids for e in encoded], dtype=np.int64)
        mask = np.array([e.attention_mask for e in encoded], dtype=np.int64)

        feed = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in self._inputs:
            # One sentence per row, so every token belongs to segment 0.
            feed["token_type_ids"] = np.zeros_like(ids)
        hidden = self._session.run(["last_hidden_state"], feed)[0]

        # Mean over unmasked tokens, then L2 normalize — the pooling the exported
        # model deliberately leaves out. Padding must be excluded or a short sentence
        # in a batch of long ones gets its meaning diluted by however many pad tokens
        # happened to sit beside it, which makes an embedding depend on its batch.
        m = mask[..., None].astype(np.float32)
        summed = (hidden * m).sum(axis=1)
        counts = np.clip(m.sum(axis=1), 1e-9, None)
        pooled = summed / counts
        norms = np.clip(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-12, None)
        return pooled / norms
