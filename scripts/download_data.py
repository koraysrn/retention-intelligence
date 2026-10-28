"""Materialize the project dataset from the committed synthetic fixture.

The original ``ecommerce_data.csv`` contains PII and cannot be redistributed.
This script copies the schema-compatible fixture to the project root and
verifies its SHA-256 checksum. If ``DATA_URL`` is set, it downloads that URL
instead (for users who have licensed access to the real data).

Usage:
    python -m scripts.download_data
    DATA_URL=https://example.com/ecommerce_data.csv python -m scripts.download_data
"""

from __future__ import annotations

import hashlib
import os
import shutil
import urllib.request
from pathlib import Path

from src.config import PROJECT_ROOT

FIXTURE = PROJECT_ROOT / "data" / "fixtures" / "ecommerce_sample.csv"
CHECKSUM = FIXTURE.with_suffix(".sha256")
TARGET = PROJECT_ROOT / "ecommerce_data.csv"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    url = os.getenv("DATA_URL", "").strip()
    if url:
        print(f"Downloading licensed dataset: {url}")
        urllib.request.urlretrieve(url, TARGET)  # noqa: S310 - explicit user-provided URL
        print(f"Wrote {TARGET} ({TARGET.stat().st_size / 1024:.1f} KiB)")
        return

    if not FIXTURE.exists():
        raise SystemExit(f"Fixture missing: {FIXTURE}. Run scripts/generate_fixture.py first.")
    expected = CHECKSUM.read_text(encoding="utf-8").split()[0] if CHECKSUM.exists() else ""
    actual = _sha256(FIXTURE)
    if expected and expected != actual:
        raise SystemExit(f"Checksum mismatch for {FIXTURE}: {actual} != {expected}")

    shutil.copyfile(FIXTURE, TARGET)
    print(f"Copied synthetic fixture -> {TARGET} (sha256={actual[:16]}...)")
    print("License: synthetic data, CC0-1.0. No real customer records are included.")


if __name__ == "__main__":
    main()
