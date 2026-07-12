from __future__ import annotations

import hashlib
import json
import platform
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

from quantlab.market_data.manifest import sha256_file


@dataclass(frozen=True)
class RegisteredArtifact:
    path: str
    sha256: str
    bytes: int


@dataclass(frozen=True)
class ExperimentRecord:
    experiment_id: str
    config_fingerprint: str
    dataset_fingerprint: str
    created_at_utc: str
    git_commit: str
    git_dirty: bool
    random_seed: int
    command: str
    python_version: str
    platform: str
    artifacts: tuple[RegisteredArtifact, ...]


def register_experiment(
    root: str | Path,
    *,
    experiment_id: str,
    config_path: str | Path,
    config_fingerprint: str,
    dataset_manifest_path: str | Path,
    dataset_fingerprint: str,
    random_seed: int,
    command: str,
    artifacts: Mapping[str, str | Path],
) -> Path:
    """Create an append-only experiment record and hash every published artifact."""
    run_dir = Path(root) / f"{experiment_id}-{config_fingerprint[:12]}"
    if run_dir.exists():
        raise FileExistsError(f"experiment record already exists: {run_dir}")
    run_dir.mkdir(parents=True)
    shutil.copy2(config_path, run_dir / "config.json")
    shutil.copy2(dataset_manifest_path, run_dir / "dataset_manifest.json")
    copied: list[RegisteredArtifact] = []
    artifact_dir = run_dir / "artifacts"
    artifact_dir.mkdir()
    for name, source in sorted(artifacts.items()):
        destination = artifact_dir / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied.append(
            RegisteredArtifact(
                str(destination.relative_to(run_dir)), sha256_file(destination), destination.stat().st_size
            )
        )
    commit = _git_output(["rev-parse", "HEAD"], default="unknown")
    dirty = bool(_git_output(["status", "--porcelain"], default=""))
    record = ExperimentRecord(
        experiment_id=experiment_id,
        config_fingerprint=config_fingerprint,
        dataset_fingerprint=dataset_fingerprint,
        created_at_utc=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        git_commit=commit,
        git_dirty=dirty,
        random_seed=random_seed,
        command=command,
        python_version=sys.version.split()[0],
        platform=platform.platform(),
        artifacts=tuple(copied),
    )
    payload = asdict(record)
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["record_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    (run_dir / "record.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return run_dir


def _git_output(arguments: list[str], default: str) -> str:
    try:
        return subprocess.run(["git", *arguments], check=True, capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return default
