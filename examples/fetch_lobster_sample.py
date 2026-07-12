from __future__ import annotations

import argparse
import shutil
import urllib.request
import zipfile
from pathlib import Path

from quantlab.market_data import build_dataset_manifest, sha256_file

DEFAULT_URL = "https://data.lobsterdata.com/info/sample/LOBSTER_SampleFile_AAPL_2012-06-21_5.zip"
LICENSE_URL = "https://data.lobsterdata.com/info/DataSamples.php"
MIRROR_ROOT = (
    "https://huggingface.co/datasets/totalorganfailure/lobster-data/resolve/main/LOBSTER_SampleFile_AAPL_2012-06-21_5"
)
EXPECTED_HASHES = {
    "AAPL_2012-06-21_34200000_57600000_message_5.csv": "8ca6fcbbf439c973d8ef74cb096a3aa340f66db816feea3ef09dcb7522bb625d",
    "AAPL_2012-06-21_34200000_57600000_orderbook_5.csv": "9a93bc2d754b5f7e02dd44cb112a19f86534212875e3b85b38ff99f6620b1f66",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the public LOBSTER AAPL level-5 demonstration sample.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/real/lobster_sample"))
    parser.add_argument("--source-url", default=DEFAULT_URL)
    parser.add_argument("--no-mirror-fallback", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    archive = args.output_dir / "LOBSTER_SampleFile_AAPL_2012-06-21_5.zip"
    download_url = args.source_url
    try:
        _download(args.source_url, archive)
        with zipfile.ZipFile(archive) as bundle:
            bundle.extractall(args.output_dir)
    except (OSError, TimeoutError, urllib.error.URLError, zipfile.BadZipFile):
        archive.unlink(missing_ok=True)
        if args.no_mirror_fallback:
            raise
        download_url = MIRROR_ROOT
        for name in EXPECTED_HASHES:
            _download(f"{MIRROR_ROOT}/{name}?download=true", args.output_dir / name)
    csv_files = sorted(args.output_dir.rglob("*.csv"))
    actual_hashes = {path.name: sha256_file(path) for path in csv_files}
    if actual_hashes != EXPECTED_HASHES:
        raise ValueError(f"LOBSTER sample checksum mismatch: {actual_hashes}")
    manifest = build_dataset_manifest(
        csv_files,
        provider="LOBSTER",
        dataset_id="AAPL-2012-06-21-level-5-sample",
        source_url=args.source_url,
        license_url=LICENSE_URL,
        download_url=download_url,
    )
    manifest_path = manifest.write(args.output_dir / "manifest.json")
    print(f"Downloaded {len(csv_files)} files; manifest={manifest_path}; fingerprint={manifest.fingerprint}")


def _download(url: str, output_path: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "quant-systems-lab/0.1"})
    with urllib.request.urlopen(request, timeout=120) as response, output_path.open("wb") as output:
        shutil.copyfileobj(response, output)


if __name__ == "__main__":
    main()
