"""Download the official SLN111 and SLN206 datasets from the totalVI paper."""

from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raw" / "totalvi_original" / "spleen_lymph_111.h5ad"
URL = "https://raw.githubusercontent.com/YosefLab/totalVI_reproducibility/master/data/spleen_lymph_111.h5ad"
SHA256 = "bb7abfe9808ac4180a9551b7f6478312d0c2080d26b69503369bc3a90d1e4235"
OUT_206 = OUT.parent / "spleen_lymph_206.h5ad"
URL_206 = "https://raw.githubusercontent.com/YosefLab/totalVI_reproducibility/master/data/spleen_lymph_206.h5ad"
SHA256_206 = "8d66161b1c2db4627c0fa60f1b7e93de28d80e6fae8001cae071b203a61f852e"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def fetch(out: Path, url: str, sha256: str) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    if not out.exists() or digest(out) != sha256:
        print(f"Downloading {out.name}", flush=True)
        urllib.request.urlretrieve(url, out)
    observed = digest(out)
    if observed != sha256:
        raise RuntimeError(f"Checksum mismatch for {out.name}: {observed}")
    print(f"Verified {out.name}: {observed}", flush=True)


def main() -> None:
    fetch(OUT, URL, SHA256)
    fetch(OUT_206, URL_206, SHA256_206)


if __name__ == "__main__":
    main()
