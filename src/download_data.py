"""Download the processed Lawlor et al. PBMC CITE-seq matrices from HCA."""

from __future__ import annotations

import hashlib
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

FILES = {
    "CZI.PBMC.RNA.matrix.Rds": (
        "https://service.azul.data.humancellatlas.org/repository/files/"
        "347257fa-4174-5741-9a9d-99ef21fa3707?catalog=dcp60&version="
        "2021-02-22T16%3A08%3A23.715288Z",
        "c6c1b29e8737085d89f2af35c23fb9ce0e189369e65061f18cfe8532bd1e80c5",
    ),
    "CZI.PBMC.ADT.matrix.Rds": (
        "https://service.azul.data.humancellatlas.org/repository/files/"
        "d3b085d0-4d42-5e0f-bf11-da491031d4b8?catalog=dcp60&version="
        "2021-02-22T16%3A08%3A23.715288Z",
        "bab0c4c2b43bb2c8c942491a370623b0685ef0dd10ce58aa45734371a8fb59e5",
    ),
    "CZI.PBMC.HTO.matrix.Rds": (
        "https://service.azul.data.humancellatlas.org/repository/files/"
        "a0e51be5-f1e2-54bc-9b92-56b025d0601d?catalog=dcp60&version="
        "2021-02-22T16%3A08%3A23.715288Z",
        "e7a4fd0fba9d9b092ca79b4e18b2d4a0763d303c10e150f2807c3e9ccb50f0dc",
    ),
    "CZI.PBMC.cell.annotations.csv": (
        "https://service.azul.data.humancellatlas.org/repository/files/"
        "773e8edd-48af-599a-8726-3f368823dc22?catalog=dcp60&version="
        "2021-02-22T16%3A08%3A23.715288Z",
        "76a6548089867687461603ed0031ba02f9203885cd8a623b7497bdceccaf7229",
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    for name, (url, expected) in FILES.items():
        path = RAW / name
        if path.exists() and sha256(path) == expected:
            print(f"Verified existing {name}")
            continue
        partial = path.with_suffix(path.suffix + ".part")
        print(f"Downloading {name}")
        with requests.get(url, stream=True, timeout=120, allow_redirects=True) as response:
            response.raise_for_status()
            with partial.open("wb") as handle:
                for chunk in response.iter_content(8 * 1024 * 1024):
                    if chunk:
                        handle.write(chunk)
        partial.replace(path)
        observed = sha256(path)
        if observed != expected:
            raise RuntimeError(f"Checksum mismatch for {name}: {observed}")
        print(f"Verified {name}: {path.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
