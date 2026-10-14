"""Server-side media roots for the MiniMax H3 Director picker.

A *media root* is a directory the Director UI may browse and reference media
from. Three kinds exist:

* ``input``  — ComfyUI's input directory (uploads land here; unchanged default).
* ``output`` — ComfyUI's output directory, i.e. everything the user already
  generated (SaveImage / SaveVideo / VideoHelperSuite …).
* custom     — any extra absolute directory the user configures, so material on
  another drive can be referenced without copying it into ``input``.

Configuration
-------------
``media_roots.json`` next to the plugin root (i.e. ``ComfyUI_MiniMaxH3_Director/
media_roots.json``)::

    {
      "roots": [
        {"id": "assets", "label": "素材盘", "path": "S:\\\\Assets"},
        {"id": "refs",   "label": "参考图库", "path": "S:\\\\Comfy-Desktop\\\\refs", "exclude": ["tmp"]}
      ],
      "exclude": ["minimax_seg_cache"]
    }

The same JSON is accepted in the ``MINIMAX_DIRECTOR_MEDIA_ROOTS`` environment
variable (handy for containers). The file is re-read when its mtime changes, so
adding a directory does not require a ComfyUI restart.

Design rules
------------
* Requests name a root by **id**, never by path, so a client cannot ask the
  server to read an arbitrary location.
* Resolution is backwards compatible: entries without ``type``/``root`` behave
  exactly as before (``input`` only), except that ``output`` and configured
  roots are tried afterwards as a convenience.
* Nothing here imports torch or ComfyUI at module level: ``folder_paths`` is
  imported lazily so this module stays unit-testable.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

log = logging.getLogger("ComfyUI-MiniMaxH3-Director.media")

CONFIG_FILENAME = "media_roots.json"
ENV_VAR = "MINIMAX_DIRECTOR_MEDIA_ROOTS"

INPUT_ID = "input"
OUTPUT_ID = "output"

#: Directory names never listed/walked: our own segment cache and other
#: bookkeeping that would otherwise flood the picker.
DEFAULT_EXCLUDED_DIRS: tuple[str, ...] = ("minimax_seg_cache",)

#: Roots whose ``/view?type=…`` works in stock ComfyUI (browser-side thumbnails).
COMFY_VIEW_TYPES: tuple[str, ...] = ("input", "output", "temp")

#: Keys checked, in order, when an entry does not say which one it uses.
_FILENAME_KEYS: tuple[str, ...] = (
    "imageFile",
    "videoFile",
    "audioFile",
    "fileName",
    "file",
    "relPath",
)

_SUBFOLDER_KEYS: tuple[str, ...] = ("subfolder", "sub_folder", "dir", "directory")

_ROOT_KEYS: tuple[str, ...] = ("type", "root", "rootId", "mediaRoot", "source")


@dataclass(frozen=True)
class MediaRoot:
    """One browsable, server-side media directory."""

    id: str
    label: str
    path: str
    comfy_type: str | None = None  # "input" / "output" / None for custom roots
    exclude: tuple[str, ...] = ()

    @property
    def builtin(self) -> bool:
        return self.comfy_type in (INPUT_ID, OUTPUT_ID)

    @property
    def view_type(self) -> str | None:
        """Value usable as ``/view?type=`` for this root, if any."""
        return self.comfy_type if self.comfy_type in COMFY_VIEW_TYPES else None

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "type": self.id,
            "viewType": self.view_type,
            "builtin": self.builtin,
            "path": self.path,
            "exclude": list(self.exclude),
        }


@dataclass
class _ConfigCache:
    mtime: float | None = None
    extra: list[dict[str, Any]] = field(default_factory=list)
    exclude: list[str] = field(default_factory=list)
    stamp: tuple = ()
    error: str = ""


_LOCK = threading.RLock()
_CACHE = _ConfigCache()


def plugin_root() -> str:
    """Absolute path of the plugin directory (``…/ComfyUI_MiniMaxH3_Director``)."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def config_path() -> str:
    return os.path.join(plugin_root(), CONFIG_FILENAME)


# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #
def _parse_config(raw: Any) -> tuple[list[dict[str, Any]], list[str]]:
    if not isinstance(raw, dict):
        return [], []
    extra: list[dict[str, Any]] = []
    for item in raw.get("roots") or []:
        if not isinstance(item, dict):
            continue
        path = str(item.get("path") or "").strip()
        if not path:
            continue
        root_id = normalize_root_id(item.get("id") or os.path.basename(path.rstrip("\\/")))
        if not root_id or root_id in (INPUT_ID, OUTPUT_ID):
            continue
        exclude = item.get("exclude")
        extra.append(
            {
                "id": root_id,
                "label": str(item.get("label") or root_id),
                "path": os.path.abspath(os.path.expanduser(path)),
                "exclude": [str(x) for x in exclude] if isinstance(exclude, (list, tuple)) else [],
            }
        )
    exclude = [str(x) for x in (raw.get("exclude") or []) if str(x).strip()]
    return extra, exclude


def _read_config() -> tuple[list[dict[str, Any]], list[str]]:
    """Read config file + env var, reusing the cache while mtime/size are stable."""
    global _CACHE
    path = config_path()
    try:
        stat = os.stat(path)
        stamp = (stat.st_mtime_ns, stat.st_size)
    except OSError:
        stamp = None

    with _LOCK:
        if stamp is not None and _CACHE.stamp == stamp:
            return _CACHE.extra, _CACHE.exclude

    extra: list[dict[str, Any]] = []
    exclude: list[str] = []
    error = ""
    if stamp is not None:
        try:
            # utf-8-sig: Notepad / PowerShell Set-Content / VS Code all happily
            # write a BOM, which plain utf-8 json.load() rejects.
            with open(path, "r", encoding="utf-8-sig") as handle:
                extra, exclude = _parse_config(json.load(handle))
        except Exception as exc:  # noqa: BLE001 - a broken config must not kill ComfyUI
            error = f"{type(exc).__name__}: {exc}"
            log.warning("MiniMax H3 Director: cannot read %s: %s", path, exc)
            extra, exclude = [], []
    env_raw = os.environ.get(ENV_VAR, "").strip()
    if env_raw:
        try:
            env_extra, env_exclude = _parse_config(json.loads(env_raw))
            known = {item["id"] for item in extra}
            extra.extend(item for item in env_extra if item["id"] not in known)
            exclude = list(dict.fromkeys([*exclude, *env_exclude]))
        except Exception as exc:  # noqa: BLE001
            error = error or f"{ENV_VAR}: {type(exc).__name__}: {exc}"
            log.warning("MiniMax H3 Director: bad %s JSON: %s", ENV_VAR, exc)

    with _LOCK:
        _CACHE = _ConfigCache(
            mtime=stamp[0] if stamp else None,
            extra=extra,
            exclude=exclude,
            stamp=stamp if stamp is not None else (),
            error=error,
        )
        return _CACHE.extra, _CACHE.exclude


def invalidate_config_cache() -> None:
    with _LOCK:
        _CACHE.stamp = ()


# --------------------------------------------------------------------------- #
# roots
# --------------------------------------------------------------------------- #
def normalize_root_id(value: Any) -> str:
    """Sanitise a client/config supplied root id (``[A-Za-z0-9_.-]``, ≤ 32 chars)."""
    text = str(value or "").strip().lower()
    if not text:
        return ""
    cleaned = "".join(ch if (ch.isalnum() or ch in "_.-") else "_" for ch in text)
    return cleaned[:32].strip("._-")


def _comfy_dir(kind: str) -> str | None:
    try:
        import folder_paths
    except Exception:  # noqa: BLE001 - ComfyUI not importable (offline tests)
        return None
    getter = (
        folder_paths.get_output_directory
        if kind == OUTPUT_ID
        else folder_paths.get_input_directory
    )
    try:
        path = getter()
    except Exception:  # noqa: BLE001
        return None
    return os.path.abspath(path) if path else None


def list_media_roots() -> list[MediaRoot]:
    """Built-in roots first (input, output), then configured custom roots."""
    extra, config_exclude = _read_config()
    roots: list[MediaRoot] = []
    for kind, label in ((INPUT_ID, "输入目录"), (OUTPUT_ID, "输出目录")):
        path = _comfy_dir(kind)
        if path:
            roots.append(
                MediaRoot(
                    id=kind,
                    label=label,
                    path=path,
                    comfy_type=kind,
                    exclude=tuple(DEFAULT_EXCLUDED_DIRS),
                )
            )
    seen = {root.id for root in roots}
    for item in extra:
        if item["id"] in seen:
            continue
        if not os.path.isdir(item["path"]):
            log.info("MiniMax H3 Director: media root %s missing: %s", item["id"], item["path"])
            continue
        seen.add(item["id"])
        roots.append(
            MediaRoot(
                id=item["id"],
                label=item["label"],
                path=item["path"],
                comfy_type=None,
                exclude=tuple(dict.fromkeys([*DEFAULT_EXCLUDED_DIRS, *config_exclude, *item["exclude"]])),
            )
        )
    return roots


def get_media_root(root_id: Any) -> MediaRoot | None:
    target = normalize_root_id(root_id)
    if not target:
        return None
    for root in list_media_roots():
        if root.id == target:
            return root
    return None


def default_root_id(preferred: Any = None) -> str:
    """``input`` unless the caller asked for a root that actually exists."""
    root = get_media_root(preferred)
    if root is not None:
        return root.id
    return INPUT_ID


def roots_for_json() -> list[dict[str, Any]]:
    return [root.to_json() for root in list_media_roots()]


def missing_roots() -> list[dict[str, str]]:
    """Configured roots whose directory does not exist (surfaced to the UI)."""
    extra, _exclude = _read_config()
    present = {root.id for root in list_media_roots()}
    return [
        {
            "id": item["id"],
            "label": item["label"],
            "path": item["path"],
            "reason": "missing-directory",
        }
        for item in extra
        if item["id"] not in present
    ]


def config_status() -> dict[str, Any]:
    """State of media_roots.json, for the picker to report configuration problems."""
    extra, _exclude = _read_config()
    path = config_path()
    with _LOCK:
        error = _CACHE.error
    return {
        "path": path,
        "exists": os.path.isfile(path),
        "error": error,
        "configured": len(extra),
    }


# --------------------------------------------------------------------------- #
# path helpers
# --------------------------------------------------------------------------- #
def is_within(root_path: str, candidate: str) -> bool:
    """True when *candidate* really lives under *root_path* (symlink aware)."""
    try:
        root_real = os.path.realpath(root_path)
        cand_real = os.path.realpath(candidate)
    except OSError:
        return False
    if os.path.commonpath([root_real, cand_real]) == root_real:
        return True
    return False


def _first_str(entry: Mapping[str, Any], keys: Iterable[str]) -> str:
    for key in keys:
        value = entry.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def split_rel(
    entry: Mapping[str, Any],
    filename_keys: Sequence[str] = _FILENAME_KEYS,
) -> tuple[str, str]:
    """Return ``(subfolder, filename)`` for a timeline entry, path-safe."""
    rel = _first_str(entry, filename_keys)
    sub = _first_str(entry, _SUBFOLDER_KEYS)
    rel = rel.replace("\\", "/").strip().lstrip("/")
    sub = sub.replace("\\", "/").strip().strip("/")
    if sub and rel.startswith(sub + "/"):
        # Some writers store subfolder + name combined; avoid duplicating it.
        rel = rel[len(sub) + 1 :]
    return sub, rel


def _candidates(root_path: str, sub: str, rel: str) -> list[str]:
    """Lookup order inside one root: as-written, then subfolder-aware, then bare name."""
    out: list[str] = []
    base_name = os.path.basename(rel)
    if sub:
        if rel:
            out.append(os.path.join(root_path, sub, rel))
        if base_name:
            out.append(os.path.join(root_path, sub, base_name))
    if rel:
        out.append(os.path.join(root_path, rel))
    if base_name:
        out.append(os.path.join(root_path, base_name))
    seen: set[str] = set()
    unique: list[str] = []
    for path in out:
        norm = os.path.normpath(path)
        if norm not in seen:
            seen.add(norm)
            unique.append(norm)
    return unique


def _search_roots(entry: Mapping[str, Any]) -> list[MediaRoot]:
    """Roots to try for *entry*, in order."""
    requested = ""
    for key in _ROOT_KEYS:
        value = entry.get(key)
        if isinstance(value, str) and value.strip():
            requested = value.strip()
            break
    if requested:
        root = get_media_root(requested)
        if root is not None:
            return [root]
        # Unknown/legacy value (e.g. an old "output" before the root existed):
        # fall through to the full order instead of failing the whole segment.
        log.debug("MiniMax H3 Director: unknown media root %r, using defaults", requested)
    return list_media_roots()


def resolve_media_path(
    entry: Mapping[str, Any] | None,
    *,
    keys: Sequence[str] | None = None,
) -> str | None:
    """Resolve a timeline media entry to an absolute path, or ``None``.

    Honours ``type``/``root`` (a root id). Without one, ``input`` is tried first
    so pre-existing workflows behave exactly as before; ``output`` and the
    configured custom roots are then tried as a convenience.
    """
    if not isinstance(entry, Mapping):
        return None
    filename_keys = tuple(keys) if keys else _FILENAME_KEYS
    sub, rel = split_rel(entry, filename_keys)
    if not rel and not sub:
        return None

    # An absolute path already on disk (third-party nodes sometimes hand us one).
    if rel and os.path.isabs(rel) and os.path.isfile(rel):
        return os.path.abspath(rel)

    for root in _search_roots(entry):
        for candidate in _candidates(root.path, sub, rel):
            if not is_within(root.path, candidate):
                continue
            if os.path.isfile(candidate):
                return candidate
    return None


def describe_missing(
    entry: Mapping[str, Any] | None,
    *,
    keys: Sequence[str] | None = None,
) -> str:
    """Human-readable "not found" message listing where we looked."""
    filename_keys = tuple(keys) if keys else _FILENAME_KEYS
    _sub, rel = split_rel(entry or {}, filename_keys)
    roots = _search_roots(entry or {})
    where = ", ".join(f"{root.id} ({root.path})" for root in roots)
    return f"{rel or '<empty>'} not found in media roots: {where}"


def media_url(entry: Mapping[str, Any] | None) -> str | None:
    """URL that serves this entry's bytes, for built-in *and* custom roots."""
    if not isinstance(entry, Mapping):
        return None
    path = resolve_media_path(entry)
    if not path:
        return None
    from urllib.parse import quote

    for root in list_media_roots():
        if not is_within(root.path, path):
            continue
        rel = os.path.relpath(path, root.path).replace("\\", "/")
        sub = os.path.dirname(rel).replace("\\", "/")
        sub = "" if sub == "." else sub
        name = os.path.basename(rel)
        if root.view_type:
            query = f"filename={quote(name)}&type={root.view_type}"
            if sub:
                query += f"&subfolder={quote(sub)}"
            return f"/view?{query}"
        return f"/minimax/director/media_file?id={quote(root.id)}&rel={quote(rel)}"
    return None
