"""Export the encoder to ONNX so the playground can run without PyTorch.

Run once: `.venv/bin/python demo-api/export_onnx.py`

**Why.** The Docker image is 2.95GB, of which about 1.02GB is a venv that is mostly
PyTorch, and the free tiers this demo would sit on give 512MB of RAM. The engine needs
torch only to turn text into 384 floats; onnxruntime does the same arithmetic in a
fraction of the space.

**What this must never touch.** The vectors an encoder produces *are* the retrieval
index. A different encoder is a different store, so ONNX embeddings may only ever be
written to the playground's own database. Running an experiment against a store built
this way would silently compare two systems while claiming to compare one. The check
at the bottom therefore reports agreement with the torch encoder rather than assuming
it: this exists to be cheap, not to be identical, and the number should be looked at.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import torch

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OUT = Path(__file__).resolve().parent / "onnx-encoder"

SAMPLES = [
    "The user lives in Melbourne.",
    "where do I live now?",
    "Your booking reference is VX-928173 and the desk is on level 3.",
    "用户住在上海浦东。",
    "The assistant recommended a desk lamp.",
]


def main() -> int:
    from sentence_transformers import SentenceTransformer

    print(f"loading {MODEL} …")
    st = SentenceTransformer(MODEL, device="cpu")
    transformer = st[0].auto_model.eval()
    tokenizer = st.tokenizer

    OUT.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(OUT)

    # The pooling layer is not exported: it is a mean over unmasked tokens followed by
    # an L2 normalize, which is four lines of numpy at load time and one less thing
    # that can disagree between frameworks. Assert the assumption rather than trust it.
    # sentence-transformers has reported this two ways across versions — a boolean
    # `pooling_mode_mean_tokens`, and a string `pooling_mode`. Accept either and fail
    # on anything else: silently pooling the wrong way produces embeddings that look
    # fine and rank differently.
    pooling = st[1].get_config_dict()
    is_mean = pooling.get("pooling_mode_mean_tokens") or pooling.get("pooling_mode") == "mean"
    assert is_mean, f"expected mean pooling, got {pooling}"
    print("pooling: mean over unmasked tokens (config verified)")

    encoded = tokenizer(SAMPLES[:1], padding=True, truncation=True, return_tensors="pt")
    args = (encoded["input_ids"], encoded["attention_mask"])
    onnx_path = OUT / "model.onnx"

    print("exporting …")
    torch.onnx.export(
        transformer,
        args,
        str(onnx_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["last_hidden_state"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "last_hidden_state": {0: "batch", 1: "seq"},
        },
        opset_version=17,
        do_constant_folding=True,
    )

    size_mb = onnx_path.stat().st_size / 1e6
    print(f"wrote {onnx_path}  ({size_mb:.0f} MB)")

    # ---------------------------------------------------------------- agreement
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from onnx_encoder import OnnxEncoder

    onnx_vecs = OnnxEncoder(OUT).encode(SAMPLES)
    torch_vecs = st.encode(SAMPLES, convert_to_numpy=True, normalize_embeddings=True)

    cos = np.sum(onnx_vecs * torch_vecs, axis=1)
    print("\nagreement with the torch encoder, per sample:")
    for text, c in zip(SAMPLES, cos, strict=True):
        print(f"  {c:.6f}  {text[:52]}")
    worst = float(cos.min())
    print(f"\nworst cosine agreement: {worst:.6f}")

    # Ranking is what the index actually does, so check that too: two encoders that
    # agree at 0.9999 per vector can still order candidates differently.
    q_onnx, q_torch = onnx_vecs[1], torch_vecs[1]
    order_onnx = np.argsort(-(onnx_vecs @ q_onnx))
    order_torch = np.argsort(-(torch_vecs @ q_torch))
    same_order = bool(np.array_equal(order_onnx, order_torch))
    print(f"same ranking for the sample query: {same_order}")

    if worst < 0.999 or not same_order:
        print("\nREFUSED: agreement is too low to serve retrieval. Nothing installed.")
        shutil.rmtree(OUT, ignore_errors=True)
        return 1

    print("\nOK — for the playground store only. Never point an experiment at this.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
