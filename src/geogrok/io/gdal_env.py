from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class GdalRuntime:
    prefix: Path
    bin_dir: Path
    data_dir: Path
    proj_dir: Path
    python_paths: tuple[Path, ...]
    library_paths: tuple[Path, ...]


def _site_packages_candidates(prefix: Path) -> tuple[Path, ...]:
    version = f"{sys.version_info.major}.{sys.version_info.minor}"
    return (
        prefix / "lib" / f"python{version}" / "site-packages",
        prefix / "lib64" / f"python{version}" / "site-packages",
    )


def discover_runtime(prefix: str | Path) -> GdalRuntime:
    resolved_prefix = Path(prefix).expanduser().resolve()
    python_paths = tuple(
        path for path in _site_packages_candidates(resolved_prefix) if path.exists()
    )
    library_paths = tuple(
        path for path in (resolved_prefix / "lib", resolved_prefix / "lib64") if path.exists()
    )
    return GdalRuntime(
        prefix=resolved_prefix,
        bin_dir=resolved_prefix / "bin",
        data_dir=resolved_prefix / "share" / "gdal",
        proj_dir=resolved_prefix / "share" / "proj",
        python_paths=python_paths,
        library_paths=library_paths,
    )


def build_environment(prefix: str | Path) -> dict[str, str]:
    runtime = discover_runtime(prefix)
    env = os.environ.copy()

    if runtime.bin_dir.exists():
        env["PATH"] = _prepend_path(runtime.bin_dir, env.get("PATH"))

    if runtime.data_dir.exists():
        env["GDAL_DATA"] = str(runtime.data_dir)

    if runtime.proj_dir.exists():
        env["PROJ_DATA"] = str(runtime.proj_dir)
        env["PROJ_LIB"] = str(runtime.proj_dir)

    if runtime.library_paths:
        joined = os.pathsep.join(str(path) for path in runtime.library_paths)
        env["LD_LIBRARY_PATH"] = _prepend_value(joined, env.get("LD_LIBRARY_PATH"))

    if runtime.python_paths:
        joined = os.pathsep.join(str(path) for path in runtime.python_paths)
        env["PYTHONPATH"] = _prepend_value(joined, env.get("PYTHONPATH"))

    return env


def activate(prefix: str | Path) -> GdalRuntime:
    runtime = discover_runtime(prefix)
    os.environ.update(build_environment(prefix))
    for path in reversed(runtime.python_paths):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    return runtime


def _prepend_path(path: Path, current: str | None) -> str:
    return _prepend_value(str(path), current)


def _prepend_value(value: str, current: str | None) -> str:
    if not current:
        return value
    parts = current.split(os.pathsep)
    if value in parts:
        return current
    return os.pathsep.join((value, current))
