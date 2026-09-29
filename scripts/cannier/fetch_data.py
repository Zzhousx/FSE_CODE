"""Fetch and checksum-verify the public CANNIER experiment artifact."""
from __future__ import annotations

import hashlib
from pathlib import Path
from urllib.request import urlretrieve

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / "results/cannier/raw/output.zip"
URL = "https://media.githubusercontent.com/media/flake-it/cannier-experiment/main/output.zip"
EXPECTED_BYTES = 354_333_950
EXPECTED_SHA256 = "07d714a7e0f57863070d1ba6631191aa6e856fac1ad3ffc92765d6c1fc56f9f1"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    DEST.parent.mkdir(parents=True, exist_ok=True)
    if not DEST.exists() or DEST.stat().st_size != EXPECTED_BYTES:
        tmp = DEST.with_suffix(".zip.part")
        urlretrieve(URL, tmp)
        size = tmp.stat().st_size
        got = sha256(tmp)
        if size != EXPECTED_BYTES or got != EXPECTED_SHA256:
            raise RuntimeError(f"artifact verification failed: bytes={size}, sha256={got}")
        tmp.replace(DEST)
    got = sha256(DEST)
    size = DEST.stat().st_size
    if size != EXPECTED_BYTES or got != EXPECTED_SHA256:
        raise RuntimeError(f"artifact verification failed: bytes={size}, sha256={got}")
    print(f"verified {DEST.name}: {size} bytes, sha256={got}")


if __name__ == "__main__":
    main()
