"""Download the public Papalexi et al. ECCITE-seq release from GEO."""

from __future__ import annotations

import hashlib
import tarfile
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "papalexi"
BASE = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE153nnn/GSE153056/suppl"

FILES = {
    "GSE153056_RAW.tar": "97bf91bfe8c04d751ade43ba6c7441366018ab26e466e55a315499909f4c5462",
    "GSE153056_ECCITE_metadata.tsv.gz": "7059b0cb857dc344e6bb53dbdc5ccadccb25bb069372ed6f45eff46b7e992a76",
}

MEMBERS = {
    "GSM4633614_ECCITE_cDNA_counts.tsv.gz",
    "GSM4633615_ECCITE_ADT_Barcodes.csv.gz",
    "GSM4633615_ECCITE_ADT_counts.tsv.gz",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    for name, expected in FILES.items():
        destination = RAW / name
        if not destination.exists() or sha256(destination) != expected:
            print(f"Downloading {name}", flush=True)
            urllib.request.urlretrieve(f"{BASE}/{name}", destination)
        observed = sha256(destination)
        if observed != expected:
            raise RuntimeError(f"Checksum mismatch for {name}: {observed}")
        print(f"Verified {name}", flush=True)

    archive = RAW / "GSE153056_RAW.tar"
    with tarfile.open(archive) as handle:
        available = {member.name: member for member in handle.getmembers()}
        missing = MEMBERS.difference(available)
        if missing:
            raise RuntimeError(f"Archive is missing expected files: {sorted(missing)}")
        for name in sorted(MEMBERS):
            destination = RAW / name
            if not destination.exists():
                handle.extract(available[name], RAW, filter="data")
                print(f"Extracted {name}", flush=True)


if __name__ == "__main__":
    main()

