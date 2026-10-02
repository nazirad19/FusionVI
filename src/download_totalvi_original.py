"""Download the official SLN111 AnnData used in the original totalVI paper."""

from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raw" / "totalvi_original" / "spleen_lymph_111.h5ad"
URL = "https://raw.githubusercontent.com/YosefLab/totalVI_reproducibility/master/data/spleen_lymph_111.h5ad"
SHA256 = "bb7abfe9808ac4180a9551b7f6478312d0c2080d26b69503369bc3a90d1e4235"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if not OUT.exists() or digest(OUT) != SHA256:
        print("Downloading official totalVI SLN111 dataset", flush=True)
        urllib.request.urlretrieve(URL, OUT)
    observed = digest(OUT)
    if observed != SHA256:
        raise RuntimeError(f"Checksum mismatch: {observed}")
    print(f"Verified {OUT.name}: {observed}", flush=True)


if __name__ == "__main__":
    main()

