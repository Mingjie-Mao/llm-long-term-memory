"""Report repository size without confusing artifacts with product source code."""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUCKETS = (
    "src",
    "tests",
    "scripts",
    "tools",
    "results",
    "docs",
    "configs",
    "public-demo",
    "demo-api",
    "root",
    "other",
)
DATA_DOC_SUFFIXES = {
    ".csv",
    ".css",
    ".html",
    ".js",
    ".json",
    ".jsonl",
    ".md",
    ".svg",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
GENERATED_SUFFIXES = {".coverage", ".lock", ".npy", ".pdf", ".svg"}


def repository_files() -> list[Path]:
    """Return tracked plus non-ignored untracked files in the working repository."""
    command = ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"]
    raw = subprocess.run(command, cwd=ROOT, check=True, capture_output=True).stdout
    return sorted(ROOT / part.decode() for part in raw.split(b"\0") if part)


def bucket(path: Path) -> str:
    relative = path.relative_to(ROOT)
    if len(relative.parts) == 1:
        return "root"
    return relative.parts[0] if relative.parts[0] in BUCKETS else "other"


def is_generated(path: Path) -> bool:
    relative = path.relative_to(ROOT)
    text = relative.as_posix()
    if path.name == "uv.lock" or path.suffix.lower() in GENERATED_SUFFIXES:
        return True
    return text.startswith(
        (
            "results/raw/",
            "results/cache/",
            "results/analysis/",
            "results/sealed/",
            "results/frozen/",
            "results/archive/",
            "docs/figures/",
        )
    )


def physical_lines(path: Path) -> int:
    data = path.read_bytes()
    if b"\0" in data:
        return 0
    return data.count(b"\n") + int(bool(data) and not data.endswith(b"\n"))


def collect() -> dict:
    rows = []
    totals = defaultdict(int)
    by_bucket: dict[str, dict[str, int]] = {name: defaultdict(int) for name in BUCKETS}
    for path in repository_files():
        if not path.is_file():
            continue
        group = bucket(path)
        size = path.stat().st_size
        lines = physical_lines(path)
        suffix = path.suffix.lower()
        generated = is_generated(path)
        row = {
            "path": path.relative_to(ROOT).as_posix(),
            "bucket": group,
            "bytes": size,
            "lines": lines,
            "python_lines": lines if suffix == ".py" else 0,
            "data_doc_lines": lines if suffix in DATA_DOC_SUFFIXES else 0,
            "generated": generated,
        }
        rows.append(row)
        for target in (totals, by_bucket[group]):
            target["files"] += 1
            target["bytes"] += size
            target["lines"] += lines
            target["python_lines"] += row["python_lines"]
            target["data_doc_lines"] += row["data_doc_lines"]
            target["generated_files"] += int(generated)
            target["generated_bytes"] += size if generated else 0
            target["generated_lines"] += lines if generated else 0
    return {"total": dict(totals), "buckets": by_bucket, "files": rows}


def human_bytes(value: int) -> str:
    units = ("B", "KiB", "MiB", "GiB")
    number = float(value)
    for unit in units:
        if number < 1024 or unit == units[-1]:
            return f"{number:.1f} {unit}"
        number /= 1024
    raise AssertionError("unreachable")


def render(data: dict) -> str:
    heading = "Category       Files       Size      Lines   Python   Data/docs  Generated(lines)"
    lines = [heading, "-" * len(heading)]
    for name in BUCKETS:
        values = data["buckets"][name]
        if not values["files"]:
            continue
        generated_ratio = values["generated_lines"] / values["lines"] if values["lines"] else 0
        lines.append(
            f"{name:14} {values['files']:5d} {human_bytes(values['bytes']):>10} "
            f"{values['lines']:10,d} {values['python_lines']:8,d} "
            f"{values['data_doc_lines']:11,d} {generated_ratio:15.1%}"
        )
    total = data["total"]
    lines += [
        "",
        f"Total repository: {total['files']:,} files, {human_bytes(total['bytes'])}, "
        f"{total['lines']:,} physical lines",
        f"Core runtime (src): {data['buckets']['src']['python_lines']:,} Python lines",
        f"Tests: {data['buckets']['tests']['python_lines']:,} Python lines",
        "Research tooling (scripts + tools): "
        f"{sum(data['buckets'][name]['python_lines'] for name in ('scripts', 'tools')):,} "
        "Python lines",
        f"Experiment artifacts (results): {data['buckets']['results']['lines']:,} lines",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = parser.parse_args()
    data = collect()
    print(json.dumps(data, indent=2) if args.json else render(data), end="\n" if args.json else "")


if __name__ == "__main__":
    main()
