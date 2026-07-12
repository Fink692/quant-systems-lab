from __future__ import annotations

import hashlib
import json
import platform
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class FileProvenance:
    path: str
    bytes: int
    sha256: str


@dataclass(frozen=True)
class DatasetManifest:
    manifest_version: str
    provider: str
    dataset_id: str
    source_url: str
    download_url: str
    license_url: str
    acquired_at_utc: str
    files: tuple[FileProvenance, ...]
    python_version: str
    platform: str

    @property
    def fingerprint(self) -> str:
        stable = {
            "manifest_version": self.manifest_version,
            "provider": self.provider,
            "dataset_id": self.dataset_id,
            "source_url": self.source_url,
            "license_url": self.license_url,
            "files": [asdict(item) for item in self.files],
        }
        payload = json.dumps(stable, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def write(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = asdict(self) | {"fingerprint": self.fingerprint}
        output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return output


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_dataset_manifest(
    paths: Iterable[str | Path],
    *,
    provider: str,
    dataset_id: str,
    source_url: str,
    license_url: str,
    download_url: str | None = None,
    acquired_at_utc: str | None = None,
) -> DatasetManifest:
    files = tuple(
        FileProvenance(path=Path(path).name, bytes=Path(path).stat().st_size, sha256=sha256_file(path))
        for path in sorted((Path(item) for item in paths), key=lambda value: value.name)
    )
    if not files:
        raise ValueError("at least one dataset file is required")
    acquired = acquired_at_utc or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    return DatasetManifest(
        manifest_version="1.0.0",
        provider=provider,
        dataset_id=dataset_id,
        source_url=source_url,
        download_url=download_url or source_url,
        license_url=license_url,
        acquired_at_utc=acquired,
        files=files,
        python_version=sys.version.split()[0],
        platform=platform.platform(),
    )
