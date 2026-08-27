"""Fetch the ONNX encoder at build time and prove it is the right one.

Run by Render's build command. Needs onnxruntime, tokenizers and numpy — no PyTorch,
which is the whole point: the torch stack is roughly 565MB of wheels against 162MB for
this path, and the free instance has 512MB of RAM.

**Why fetch rather than commit.** The model is 87MB. Committing it means Git LFS or an
87MB blob in every clone; exporting it at build means installing PyTorch to produce a
file whose entire purpose is to avoid PyTorch.

**Why the check is not optional.** A build that downloads a model and starts serving is
trusting a URL. The download could be the wrong revision, a different export with a
different signature, or truncated. The build environment cannot recompute the reference
— torch is not installed — so `reference_embeddings.json` carries vectors produced
locally by the same torch encoder every published number was measured with, and this
script asserts the fetched model reproduces them. Below threshold, the build fails
rather than serving embeddings that are subtly not the ones the design was measured on.
"""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

import numpy as np

REPO = "sentence-transformers/all-MiniLM-L6-v2"
BASE = f"https://huggingface.co/{REPO}/resolve/main"
FILES = {
    "model.onnx": f"{BASE}/onnx/model.onnx",
    "tokenizer.json": f"{BASE}/tokenizer.json",
    "tokenizer_config.json": f"{BASE}/tokenizer_config.json",
}
HERE = Path(__file__).resolve().parent
OUT = HERE / "onnx-encoder"
MIN_COSINE = 0.999


def fetch() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, url in FILES.items():
        target = OUT / name
        if target.exists() and target.stat().st_size > 0:
            print(f"  have {name} ({target.stat().st_size / 1e6:.1f} MB)")
            continue
        print(f"  fetching {name} …")
        urllib.request.urlretrieve(url, target)
        print(f"    {target.stat().st_size / 1e6:.1f} MB")


def verify() -> int:
    sys.path.insert(0, str(HERE))
    from onnx_encoder import OnnxEncoder

    reference = json.loads((HERE / "reference_embeddings.json").read_text(encoding="utf-8"))
    expected = np.array(reference["vectors"], dtype=np.float32)

    encoder = OnnxEncoder(OUT)
    if encoder.dim != reference["dim"]:
        print(f"FAIL: dim {encoder.dim} != {reference['dim']}")
        return 1

    got = encoder.encode(reference["texts"])
    cos = np.sum(got * expected, axis=1)
    for text, c in zip(reference["texts"], cos, strict=True):
        print(f"  {c:.6f}  {text[:52]}")

    worst = float(cos.min())
    print(f"\nworst agreement with the reference encoder: {worst:.6f}")
    if worst < MIN_COSINE:
        print(f"FAIL: below {MIN_COSINE}. Refusing to serve a different encoder.")
        return 1

    # Agreement per vector is not agreement on order, and order is what the index
    # returns. Check that the fetched model ranks the reference set the same way.
    query = 1
    if not np.array_equal(
        np.argsort(-(got @ got[query])), np.argsort(-(expected @ expected[query]))
    ):
        print("FAIL: same vectors, different ranking.")
        return 1
    print("ranking matches the reference")
    return 0


if __name__ == "__main__":
    print(f"encoder <- {REPO}")
    fetch()
    raise SystemExit(verify())
