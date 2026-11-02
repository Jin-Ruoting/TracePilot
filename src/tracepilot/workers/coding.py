from __future__ import annotations

import ast
import difflib
import hashlib
import os
import shutil
from pathlib import Path
from typing import Any

from ..research.models import Variant, canonical, digest

EXCLUDED = {".git", ".venv", ".venv-research", "venv", "__pycache__", ".cache", ".pytest_cache",
            "node_modules", ".env", ".ssh", ".aws"}


def file_manifest(root: Path, *, exclude_top: str = "") -> dict[str, str]:
    root = root.resolve()
    manifest: dict[str, str] = {}
    size = 0
    paths = []
    for current, directories, filenames in os.walk(root, followlinks=False):
        base = Path(current)
        directories[:] = sorted(name for name in directories if name not in EXCLUDED
                                and not name.startswith(".env.")
                                and not (base == root and name == exclude_top))
        for name in directories:
            if (base / name).is_symlink():
                raise ValueError(f"source symlinks require explicit materialization: {base / name}")
        paths.extend(base / name for name in filenames)
    for path in sorted(paths):
        relative = path.relative_to(root)
        if any(part in EXCLUDED or part.startswith(".env.") for part in relative.parts):
            continue
        if exclude_top and relative.parts[0] == exclude_top:
            continue
        if path.is_symlink():
            raise ValueError(f"source symlinks require explicit materialization: {relative}")
        if not path.is_file():
            continue
        size += path.stat().st_size
        if size > 25_000_000:
            raise ValueError("code snapshot exceeds 25 MB; keep datasets and checkpoints outside the code workspace")
        manifest[relative.as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    if not manifest:
        raise ValueError("empty code workspace")
    return manifest


def snapshot_workspace(source: Path, target: Path) -> dict[str, str]:
    source, target = source.resolve(), target.resolve()
    if source == target:
        raise ValueError("snapshot destination must differ from source")
    exclude = target.relative_to(source).parts[0] if target.is_relative_to(source) else ""
    manifest = file_manifest(source, exclude_top=exclude)
    if target.exists():
        if file_manifest(target) != manifest:
            raise ValueError("source or existing snapshot changed; create a new project revision")
        return manifest
    target.mkdir(parents=True)
    for relative, expected in manifest.items():
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / relative, destination)
        if hashlib.sha256(destination.read_bytes()).hexdigest() != expected:
            raise ValueError("source changed while taking the snapshot")
    return manifest


class CodingWorker:
    """Materialize an explicitly supplied or model-proposed patch in an isolated workspace."""

    def prepare(self, source: Path, target: Path, variant: Variant) -> dict[str, Any]:
        if target.exists():
            raise FileExistsError(target)
        shutil.copytree(source, target)
        patch: list[str] = []
        for change in variant.changes:
            destination = (target / change["path"]).resolve()
            if not destination.is_relative_to(target.resolve()):
                raise ValueError("patch escapes the isolated workspace")
            previous = destination.read_bytes() if destination.exists() else b""
            if hashlib.sha256(previous).hexdigest() != change["expected_sha256"]:
                raise ValueError(f"stale patch: {change['path']}")
            if destination.suffix == ".py":
                ast.parse(change["content"], filename=change["path"])
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(change["content"])
            patch.extend(difflib.unified_diff(previous.decode().splitlines(True),
                                             change["content"].splitlines(True),
                                             fromfile="a/" + change["path"], tofile="b/" + change["path"]))
        manifest = file_manifest(target)
        result = {"variant_id": variant.variant_id, "code_hash": digest(manifest),
                  "files": manifest, "patch": "".join(patch), "patch_origin": "supplied_spec"}
        target.parent.joinpath(variant.variant_id + ".provenance.json").write_text(canonical(result) + "\n")
        return result
