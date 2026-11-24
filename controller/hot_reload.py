"""In-process reloading helpers for the web controller.

Two tiers are supported:

* calibration files (``calibration/*.json`` and ``*.npz``) are re-read and
  swapped into the running controller, so constants take effect immediately;
* Python sources are re-imported with :func:`importlib.reload` and the web
  application is rebuilt, so code changes take effect without restarting the
  Python process.

The module deliberately avoids importing any controller code at import time so
that it can survive reloads of the rest of the package.
"""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

WATCHED_SUFFIXES = frozenset({".py", ".json", ".npz"})
_IGNORED_PARTS = frozenset(
    {
        "__pycache__",
        ".git",
        ".hg",
        ".svn",
        ".venv",
        "venv",
        "node_modules",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
    }
)

# Parents before children so that reloaded modules pick up newly reloaded names.
# A second pass in ``reload_modules`` covers any remaining cross-references and
# modules that were added after this list was written.
_RELOAD_ORDER = (
    "controller.config.settings",
    "controller.motion.klipper_client",
    "controller.network.moonraker",
    "controller.vision.tubeLocator.tube_locator",
    "controller.vision.qr_decoder",
    "controller.vision.camera_preview",
    "controller.workflow.state_machine",
    "controller.web.errors",
    "controller.web.events",
    "controller.web.actions",
    "controller.web.controller",
    "controller.web.runtime",
    "controller.web.server",
)
_SKIP_MODULES = frozenset({"controller.hot_reload", "__main__", "controller.main", "controller.__main__"})


@dataclass(frozen=True)
class ChangeSet:
    """Files that changed since the previous poll."""

    config_paths: tuple[Path, ...] = ()
    code_paths: tuple[Path, ...] = ()

    @property
    def empty(self) -> bool:
        return not self.config_paths and not self.code_paths

    @property
    def paths(self) -> tuple[Path, ...]:
        return self.code_paths + self.config_paths

    def merged(self, other: "ChangeSet") -> "ChangeSet":
        return ChangeSet(
            config_paths=tuple(sorted(set(self.config_paths) | set(other.config_paths))),
            code_paths=tuple(sorted(set(self.code_paths) | set(other.code_paths))),
        )

    def describe(self) -> str:
        names = [path.name for path in self.paths]
        return ", ".join(dict.fromkeys(names))


class SourceWatcher:
    """Poll calibration and source files for modifications."""

    def __init__(self, config_dirs: Iterable[Path], source_root: Path) -> None:
        self._config_roots = tuple(Path(directory).resolve() for directory in config_dirs)
        self._source_root = Path(source_root).resolve()
        self._roots = (self._source_root / "controller", *self._config_roots)
        self._snapshot = self._scan()

    def poll(self) -> ChangeSet:
        current = self._scan()
        changed = {
            Path(path)
            for path, mtime in current.items()
            if self._snapshot.get(path) != mtime
        }
        changed.update(Path(path) for path in self._snapshot.keys() - current.keys())
        self._snapshot = current

        config_paths: list[Path] = []
        code_paths: list[Path] = []
        for path in sorted(changed):
            if self._is_source_asset(path):
                code_paths.append(path)
            else:
                config_paths.append(path)
        return ChangeSet(config_paths=tuple(config_paths), code_paths=tuple(code_paths))

    def _is_source_asset(self, path: Path) -> bool:
        """Return True for Python and for assets baked into vision code."""
        try:
            relative = path.relative_to(self._source_root)
        except ValueError:
            return False
        return relative.parts[0] == "controller" and (
            path.suffix == ".py" or path.suffix in {".json", ".npz"}
        )

    def _scan(self) -> dict[str, int]:
        snapshot: dict[str, int] = {}
        for root in self._roots:
            if not root.is_dir():
                continue
            for path in root.rglob("*"):
                if path.suffix not in WATCHED_SUFFIXES or not path.is_file():
                    continue
                if _IGNORED_PARTS.intersection(path.parts):
                    continue
                try:
                    snapshot[str(path)] = path.stat().st_mtime_ns
                except OSError:
                    continue
        return snapshot


def reload_modules() -> tuple[str, ...]:
    """Re-execute every imported controller module in place.

    Raises :class:`RuntimeError` if a module fails to import; the caller is
    expected to keep the previous application alive in that case.
    """
    names = _module_names()
    reloaded: list[str] = []
    for _ in range(2):
        for name in names:
            module = sys.modules.get(name)
            if module is None:
                continue
            try:
                importlib.reload(module)
            except Exception as exc:  # pragma: no cover - defensive boundary
                raise RuntimeError(f"Could not reload {name}: {exc}") from exc
            if name not in reloaded:
                reloaded.append(name)
    return tuple(reloaded)


def touched_vision(paths: Iterable[Path], source_root: Path) -> bool:
    """Return True when a change affects the camera capture thread."""
    source_root = Path(source_root).resolve()
    for path in paths:
        try:
            relative = Path(path).resolve().relative_to(source_root)
        except ValueError:
            continue
        if len(relative.parts) > 1 and relative.parts[0] == "controller" and relative.parts[1] == "vision":
            return True
    return False


def _module_names() -> list[str]:
    discovered = {
        name
        for name in sys.modules
        if name.startswith("controller.") and name not in _SKIP_MODULES
    }
    ordered = [name for name in _RELOAD_ORDER if name in discovered]
    remaining = sorted(
        discovered.difference(ordered).difference(_RELOAD_ORDER),
        key=lambda name: (name.count("."), name),
    )
    return ordered + remaining
