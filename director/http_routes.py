"""HTTP routes for MiniMax H3 Director (chunked video upload)."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import subprocess
import uuid

import folder_paths
from aiohttp import web
from server import PromptServer

from ..lib.media_roots import (
    INPUT_ID as INPUT_ROOT_ID,
    config_status,
    default_root_id,
    get_media_root,
    is_within,
    list_media_roots,
    missing_roots,
    normalize_root_id,
    roots_for_json,
)

log = logging.getLogger("ComfyUI-MiniMaxH3-Director.director")

CHUNK_ROOT = os.path.join(folder_paths.get_temp_directory(), "minimax_upload_chunks")
REF_AUDIO_CHUNK_ROOT = os.path.join(folder_paths.get_temp_directory(), "minimax_ref_audio_chunks")
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff"}
VIDEO_EXTS = {".mp4", ".mov", ".webm", ".mkv", ".avi", ".m4v", ".mpg", ".mpeg", ".mts", ".ts"}
AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".wma"}
_WIN_ILLEGAL = re.compile(r'[<>:"/\\|?*\x00-\x1f]+')
_WIN_RESERVED = re.compile(r"^(con|prn|aux|nul|com[1-9]|lpt[1-9])$", re.I)
_SAFE_EXT = re.compile(r"\.[A-Za-z0-9]{1,8}$")
_ROUTES_REGISTERED = False


def _safe_basename(name: str) -> str:
    """Keep CJK names; only strip path pieces and Windows-illegal characters."""
    base = os.path.basename(str(name or "upload.bin").replace("\\", "/"))
    stem, ext = os.path.splitext(base)
    ext = ext.lower()
    if not _SAFE_EXT.fullmatch(ext):
        ext = ".bin"
    if ext == ".jpeg":
        ext = ".jpg"
    stem = _WIN_ILLEGAL.sub("_", stem).rstrip(" .")[:80]
    if not stem or _WIN_RESERVED.match(stem):
        stem = "upload"
    return f"{stem}{ext}"


def _get_media_exts(kind: str) -> set[str]:
    kind = str(kind or "").strip().lower()
    if kind == "image":
        return IMAGE_EXTS
    if kind == "video":
        return VIDEO_EXTS
    if kind == "audio":
        return AUDIO_EXTS
    if kind == "reference_audio":
        return AUDIO_EXTS | VIDEO_EXTS
    raise ValueError("kind must be image, video, audio or reference_audio")


def _peek_image_size(path: str) -> tuple[int, int]:
    """Read width/height from the image header without decoding pixels."""
    try:
        from PIL import Image

        with Image.open(path) as im:
            w, h = im.size
            return int(w or 0), int(h or 0)
    except Exception:
        return 0, 0


def _media_kind_for_ext(ext: str) -> str:
    if ext in VIDEO_EXTS:
        return "video"
    if ext in AUDIO_EXTS:
        return "audio"
    return "image"


def _iter_root_items(root, exts: set[str]) -> list[dict]:
    """Walk one media root; no stat-heavy size probing here (see _decorate_items)."""
    items: list[dict] = []
    excluded = set(root.exclude or ())
    for current, dirs, files in os.walk(root.path):
        dirs[:] = [d for d in dirs if not d.startswith(".") and d not in excluded]
        for name in files:
            if name.startswith("."):
                continue
            ext = os.path.splitext(name)[1].lower()
            if ext not in exts:
                continue
            abs_path = os.path.join(current, name)
            try:
                stat = os.stat(abs_path)
            except OSError:
                continue
            rel_path = os.path.relpath(abs_path, root.path).replace("\\", "/")
            if rel_path.startswith(".."):
                continue
            subfolder = os.path.dirname(rel_path).replace("\\", "/")
            if subfolder == ".":
                subfolder = ""
            items.append(
                {
                    "name": name,
                    "fileName": name,
                    "relPath": rel_path,
                    "subfolder": subfolder,
                    "type": root.id,
                    "root": root.id,
                    "rootLabel": root.label,
                    "viewType": root.view_type,
                    "modified": float(stat.st_mtime),
                    "size": int(stat.st_size),
                    "width": 0,
                    "height": 0,
                    "mediaKind": _media_kind_for_ext(ext),
                    "path": abs_path,
                }
            )
    return items


def _decorate_items(items: list[dict]) -> list[dict]:
    """Fill width/height for the page we are about to return (header reads only)."""
    from ..lib.video_io import peek_video_size

    for item in items:
        ext = os.path.splitext(item["name"])[1].lower()
        path = item.pop("path", "")
        if not path:
            continue
        if ext in IMAGE_EXTS:
            item["width"], item["height"] = _peek_image_size(path)
        elif ext in VIDEO_EXTS:
            try:
                item["width"], item["height"] = peek_video_size(path)
            except Exception:
                item["width"], item["height"] = 0, 0
    return items


def _list_media(
    kind: str,
    *,
    roots: list[str] | None = None,
    limit: int = 400,
    offset: int = 0,
    query: str = "",
) -> dict:
    """List media across media roots.

    ``roots=None`` means "every configured root" (input, output, custom).
    Sizes are probed only for the returned page, and every root is walked
    before sorting so the newest files win regardless of which root they sit in.
    """
    exts = _get_media_exts(kind)
    all_roots = list_media_roots()
    wanted = [normalize_root_id(r) for r in (roots or []) if str(r or "").strip()]
    if wanted:
        selected = [root for root in all_roots if root.id in wanted]
    else:
        selected = list(all_roots)

    needle = str(query or "").strip().lower()
    found: list[dict] = []
    for root in selected:
        for item in _iter_root_items(root, exts):
            if needle and needle not in item["relPath"].lower():
                continue
            found.append(item)
    found.sort(key=lambda item: (-item["modified"], item["root"], item["relPath"]))

    total = len(found)
    offset = max(0, int(offset))
    limit = max(1, min(int(limit), 5000))
    page = found[offset : offset + limit]
    return {
        "items": _decorate_items(page),
        "total": total,
        "offset": offset,
        "limit": limit,
        "truncated": offset + len(page) < total,
        "roots": [root.id for root in selected],
    }


def _list_input_media(kind: str) -> list[dict]:
    """Legacy single-root listing (input only, no limit) kept for callers."""
    return _list_media(kind, roots=[INPUT_ROOT_ID], limit=5000)["items"]


async def minimax_upload_video_chunk(request):
    try:
        post = await request.post()
    except Exception as exc:
        return web.Response(status=400, text=f"Invalid upload: {exc}")

    upload_id = str(post.get("upload_id") or "").strip()
    filename = _safe_basename(post.get("filename"))
    chunk_field = post.get("chunk")
    if not upload_id or chunk_field is None:
        return web.Response(status=400, text="Missing upload_id or chunk.")

    if ".." in upload_id or "/" in upload_id or "\\" in upload_id:
        return web.Response(status=400, text="Invalid upload_id.")

    try:
        chunk_index = int(post.get("chunk_index", 0))
        total_chunks = int(post.get("total_chunks", 1))
    except (TypeError, ValueError):
        return web.Response(status=400, text="Invalid chunk index.")

    if total_chunks < 1 or chunk_index < 0 or chunk_index >= total_chunks:
        return web.Response(status=400, text="Chunk index out of range.")

    session_dir = os.path.join(CHUNK_ROOT, upload_id)
    os.makedirs(session_dir, exist_ok=True)
    part_path = os.path.join(session_dir, f"{chunk_index:06d}.part")

    with open(part_path, "wb") as out:
        while True:
            block = chunk_field.file.read(1024 * 1024)
            if not block:
                break
            out.write(block)

    if chunk_index + 1 < total_chunks:
        return web.json_response({"status": "ok", "chunk_index": chunk_index})

    input_dir = folder_paths.get_input_directory()
    out_path = os.path.join(input_dir, filename)
    if os.path.exists(out_path):
        stem, ext = os.path.splitext(filename)
        for n in range(1, 1000):
            candidate = f"{stem}_{n}{ext}"
            candidate_path = os.path.join(input_dir, candidate)
            if not os.path.exists(candidate_path):
                out_path = candidate_path
                filename = candidate
                break

    with open(out_path, "wb") as out:
        for i in range(total_chunks):
            part = os.path.join(session_dir, f"{i:06d}.part")
            if not os.path.isfile(part):
                shutil.rmtree(session_dir, ignore_errors=True)
                return web.Response(status=400, text=f"Missing chunk {i}.")
            with open(part, "rb") as src:
                shutil.copyfileobj(src, out)

    shutil.rmtree(session_dir, ignore_errors=True)
    log.info("MiniMax H3 Director uploaded video to input/: %s", filename)
    return web.json_response({"name": filename, "subfolder": "", "type": "input"})


def _reference_audio_result(path: str, *, reused: bool, source_kind: str) -> dict:
    name = os.path.basename(path)
    return {
        "name": name,
        "fileName": name,
        "relPath": name,
        "subfolder": "",
        "type": "input",
        "reused": bool(reused),
        "sourceKind": source_kind,
    }


def _files_identical(first: str, second: str) -> bool:
    """Match ComfyUI upload dedupe without assigning content-derived filenames."""
    try:
        if os.path.getsize(first) != os.path.getsize(second):
            return False
        with open(first, "rb") as left, open(second, "rb") as right:
            while True:
                left_block = left.read(4 * 1024 * 1024)
                right_block = right.read(4 * 1024 * 1024)
                if left_block != right_block:
                    return False
                if not left_block:
                    return True
    except OSError:
        return False


def _place_in_input_like_comfy_upload(temp_path: str, filename: str) -> tuple[str, bool]:
    """Use ComfyUI's non-overwrite rule: reuse identical, otherwise append ` (n)`."""
    input_dir = folder_paths.get_input_directory()
    filename = _safe_basename(filename)
    stem, ext = os.path.splitext(filename)
    candidate_name = filename
    index = 1
    while True:
        candidate_path = os.path.join(input_dir, candidate_name)
        if not os.path.exists(candidate_path):
            os.replace(temp_path, candidate_path)
            return candidate_path, False
        if _files_identical(candidate_path, temp_path):
            os.remove(temp_path)
            return candidate_path, True
        candidate_name = f"{stem} ({index}){ext}"
        index += 1


def _prepare_reference_audio(source_path: str, display_name: str) -> dict:
    """Extract a video's first audio stream and place it directly in input/."""
    if not os.path.isfile(source_path) or os.path.getsize(source_path) <= 0:
        raise ValueError("Reference audio source is empty or missing.")
    ext = os.path.splitext(display_name or source_path)[1].lower()
    if ext not in VIDEO_EXTS:
        raise ValueError("Selected source is not a supported video.")

    safe_name = _safe_basename(display_name or os.path.basename(source_path))
    safe_stem = os.path.splitext(safe_name)[0] or "reference_audio"
    output_name = f"{safe_stem}.flac"
    output_dir = folder_paths.get_input_directory()

    from ..lib.audio_io import _ffmpeg_bin

    ffmpeg = _ffmpeg_bin()
    if not ffmpeg:
        raise RuntimeError("ffmpeg is unavailable; cannot extract audio from video.")
    tmp_path = os.path.join(output_dir, f".minimax_ref_audio_{uuid.uuid4().hex}.flac")
    args = [
        ffmpeg,
        "-v",
        "error",
        "-nostdin",
        "-i",
        source_path,
        "-map",
        "0:a:0",
        "-vn",
        "-c:a",
        "flac",
        "-compression_level",
        "5",
        "-y",
        tmp_path,
    ]
    try:
        result = subprocess.run(args, capture_output=True, check=False)
        if result.returncode != 0 or not os.path.isfile(tmp_path) or os.path.getsize(tmp_path) <= 0:
            error = (result.stderr or b"").decode("utf-8", errors="replace").strip()
            raise RuntimeError(error or "The selected video has no decodable audio stream.")
        output_path, reused = _place_in_input_like_comfy_upload(tmp_path, output_name)
    finally:
        try:
            if os.path.isfile(tmp_path):
                os.remove(tmp_path)
        except OSError:
            pass
    return _reference_audio_result(output_path, reused=reused, source_kind="video")


async def minimax_extract_reference_audio(request):
    """Extract an existing input video's audio immediately into input/."""
    try:
        body = await request.json()
        video_file = str(body.get("videoFile") or body.get("relPath") or "").strip()
        if not video_file:
            return web.Response(status=400, text="Missing videoFile.")
        from ..lib.video_io import resolve_video_path

        clip = {
            "videoFile": video_file,
            "fileName": str(body.get("fileName") or os.path.basename(video_file)),
            "subfolder": str(body.get("subfolder") or ""),
            "type": str(body.get("type") or "input"),
        }
        source_path = resolve_video_path(clip)
        if os.path.splitext(source_path)[1].lower() not in VIDEO_EXTS:
            return web.Response(status=400, text="Selected source is not a supported video.")
        result = await asyncio.to_thread(
            _prepare_reference_audio,
            source_path,
            clip["fileName"] or os.path.basename(source_path),
        )
        return web.json_response(result)
    except Exception as exc:
        log.warning("MiniMax H3 Director reference audio extraction failed: %s", exc)
        return web.Response(status=400, text=str(exc))


async def minimax_prepare_reference_audio_chunk(request):
    """Receive large local audio/video; store audio or extract video audio into input/."""
    session_dir = ""
    try:
        post = await request.post()
        upload_id = str(post.get("upload_id") or "").strip()
        filename = _safe_basename(post.get("filename"))
        chunk_field = post.get("chunk")
        if not re.fullmatch(r"[A-Za-z0-9_-]{8,80}", upload_id) or chunk_field is None:
            return web.Response(status=400, text="Invalid reference audio upload.")
        try:
            chunk_index = int(post.get("chunk_index", 0))
            total_chunks = int(post.get("total_chunks", 1))
        except (TypeError, ValueError):
            return web.Response(status=400, text="Invalid chunk index.")
        if total_chunks < 1 or chunk_index < 0 or chunk_index >= total_chunks:
            return web.Response(status=400, text="Chunk index out of range.")
        source_ext = os.path.splitext(filename)[1].lower()
        if source_ext not in AUDIO_EXTS | VIDEO_EXTS:
            return web.Response(status=400, text="Unsupported reference audio source format.")

        session_dir = os.path.join(REF_AUDIO_CHUNK_ROOT, upload_id)
        os.makedirs(session_dir, exist_ok=True)
        part_path = os.path.join(session_dir, f"{chunk_index:06d}.part")
        with open(part_path, "wb") as out:
            while True:
                block = chunk_field.file.read(1024 * 1024)
                if not block:
                    break
                out.write(block)
        if chunk_index + 1 < total_chunks:
            response = web.json_response({"status": "ok", "chunk_index": chunk_index})
            session_dir = ""
            return response

        source_path = os.path.join(session_dir, filename)
        with open(source_path, "wb") as out:
            for index in range(total_chunks):
                part = os.path.join(session_dir, f"{index:06d}.part")
                if not os.path.isfile(part):
                    raise ValueError(f"Missing chunk {index}.")
                with open(part, "rb") as src:
                    shutil.copyfileobj(src, out)
        if source_ext in AUDIO_EXTS:
            output_path, reused = await asyncio.to_thread(
                _place_in_input_like_comfy_upload,
                source_path,
                filename,
            )
            result = _reference_audio_result(output_path, reused=reused, source_kind="audio")
        else:
            result = await asyncio.to_thread(_prepare_reference_audio, source_path, filename)
        return web.json_response(result)
    except Exception as exc:
        log.warning("MiniMax H3 Director local reference audio preparation failed: %s", exc)
        return web.Response(status=400, text=str(exc))
    finally:
        if session_dir:
            shutil.rmtree(session_dir, ignore_errors=True)


async def minimax_probe_video(request):
    try:
        if request.can_read_body and request.content_type == "application/json":
            body = await request.json()
        else:
            body = dict(request.query)
    except Exception as exc:
        return web.Response(status=400, text=f"Invalid request: {exc}")

    video_file = str(body.get("videoFile") or body.get("video_file") or "").strip()
    if not video_file:
        return web.Response(status=400, text="Missing videoFile.")

    from ..lib.video_io import probe_video_clip

    clip = {
        "videoFile": video_file,
        "fileName": os.path.basename(video_file),
        "subfolder": str(body.get("subfolder") or "").strip(),
        "type": str(body.get("type") or "input").strip() or "input",
    }
    try:
        info = probe_video_clip(clip)
    except Exception as exc:
        log.warning("MiniMax H3 Director video probe failed: %s", exc)
        return web.Response(status=400, text=str(exc))
    return web.json_response(info)


async def minimax_list_input_media(request):
    """List media files across the configured media roots (input / output / custom).

    Query params: ``kind`` (required), ``root`` (comma separated root ids, default
    all), ``q`` (case-insensitive substring filter), ``limit`` (default 400),
    ``offset`` (paging).
    """
    try:
        kind = str(request.query.get("kind") or "").strip().lower()
        if not kind:
            return web.Response(status=400, text="Missing kind.")
        roots_param = request.query.get("root") or request.query.get("roots") or ""
        roots = [part for part in re.split(r"[,\s]+", roots_param) if part]
        try:
            limit = int(request.query.get("limit") or 400)
        except (TypeError, ValueError):
            limit = 400
        try:
            offset = int(request.query.get("offset") or 0)
        except (TypeError, ValueError):
            offset = 0
        query = str(request.query.get("q") or "")
        # Walking output/ can touch thousands of files: keep it off the event loop.
        payload = await asyncio.to_thread(
            _list_media, kind, roots=roots, limit=limit, offset=offset, query=query
        )
    except ValueError as exc:
        return web.Response(status=400, text=str(exc))
    except Exception as exc:
        log.warning("MiniMax H3 Director list media failed: %s", exc)
        return web.Response(status=500, text=str(exc))
    return web.json_response(payload)


async def minimax_media_roots(request):
    """Describe the media roots the picker may browse (id / label / view type)."""
    try:
        input_root = get_media_root(INPUT_ROOT_ID)
        payload = {
            "default": default_root_id(),
            "roots": roots_for_json(),
            "inputDir": input_root.path if input_root else "",
            "missing": missing_roots(),
            "config": config_status(),
        }
    except Exception as exc:
        log.warning("MiniMax H3 Director media roots failed: %s", exc)
        return web.Response(status=500, text=str(exc))
    return web.json_response(payload)


async def minimax_media_file(request):
    """Serve one file from any configured media root (used for custom roots).

    ComfyUI's own ``/view`` only understands input/output/temp, so the picker
    falls back to this route for user-configured directories. The client passes a
    root **id** plus a relative path; absolute paths and traversal are rejected.
    """
    root = get_media_root(request.query.get("id"))
    if root is None:
        return web.Response(status=404, text="Unknown media root.")
    rel = str(request.query.get("rel") or "").replace("\\", "/").strip().lstrip("/")
    if not rel:
        return web.Response(status=400, text="Missing rel.")
    if os.path.isabs(rel) or ".." in rel.split("/"):
        return web.Response(status=400, text="Invalid rel path.")
    path = os.path.normpath(os.path.join(root.path, rel.replace("/", os.sep)))
    if not is_within(root.path, path) or not os.path.isfile(path):
        return web.Response(status=404, text="File not found.")
    response = web.FileResponse(path)
    response.headers["Cache-Control"] = "private, max-age=60"
    return response


async def minimax_detect_shots(request):
    """Detect shot boundaries with PySceneDetect; return logical cut frames."""
    try:
        body = await request.json()
    except Exception as exc:
        return web.Response(status=400, text=f"Invalid JSON: {exc}")

    from ..lib.shot_detect import (
        detect_timeline_shot_cuts,
        scenedetect_available,
        scenedetect_install_hint,
    )

    if not scenedetect_available():
        return web.Response(
            status=400,
            text=(
                "PySceneDetect is not installed in ComfyUI's Python "
                f"({__import__('sys').executable}). "
                f"Run: {scenedetect_install_hint()}"
            ),
        )

    try:
        frame_rate = float(body.get("frameRate") or body.get("frame_rate") or 24)
    except (TypeError, ValueError):
        frame_rate = 24.0
    try:
        total_frames = int(body.get("totalFrames") or body.get("total_frames") or 0)
    except (TypeError, ValueError):
        return web.Response(status=400, text="Invalid totalFrames.")

    sensitivity = str(body.get("sensitivity") or "medium").strip().lower()
    try:
        min_shot_frames = int(body.get("minShotFrames") or body.get("min_shot_frames") or 12)
    except (TypeError, ValueError):
        min_shot_frames = 12

    clips_in = body.get("clips")
    clips: list[dict] = []
    if isinstance(clips_in, list) and clips_in:
        for item in clips_in:
            if not isinstance(item, dict):
                continue
            video_file = str(item.get("videoFile") or item.get("video_file") or "").strip()
            if not video_file:
                continue
            clips.append(
                {
                    "videoFile": video_file,
                    "fileName": os.path.basename(video_file),
                    "subfolder": str(item.get("subfolder") or "").strip(),
                    "type": str(item.get("type") or "input").strip() or "input",
                    "logicalStart": item.get("logicalStart", item.get("logical_start", 0)),
                    "logicalEnd": item.get("logicalEnd", item.get("logical_end", total_frames)),
                    "nativeFps": item.get("nativeFps", item.get("native_fps")),
                }
            )
    else:
        video_file = str(body.get("videoFile") or body.get("video_file") or "").strip()
        if not video_file:
            return web.Response(status=400, text="Missing clips[] or videoFile.")
        clips.append(
            {
                "videoFile": video_file,
                "fileName": os.path.basename(video_file),
                "subfolder": str(body.get("subfolder") or "").strip(),
                "type": str(body.get("type") or "input").strip() or "input",
                "logicalStart": 0,
                "logicalEnd": total_frames,
                "nativeFps": body.get("nativeFps", body.get("native_fps")),
            }
        )

    if total_frames <= 0:
        return web.Response(status=400, text="totalFrames must be > 0.")

    try:
        result = detect_timeline_shot_cuts(
            clips,
            frame_rate=frame_rate,
            total_frames=total_frames,
            sensitivity=sensitivity,
            min_shot_frames=min_shot_frames,
        )
    except ImportError as exc:
        return web.Response(status=400, text=str(exc))
    except Exception as exc:
        log.warning("MiniMax H3 Director shot detect failed: %s", exc)
        return web.Response(status=400, text=str(exc))

    return web.json_response(result)


async def minimax_first_pass_cache_status(request):
    """Compare stored first-pass metadata with the Director's current inputs."""
    try:
        body = await request.json()
    except Exception as exc:
        return web.Response(status=400, text=f"Invalid JSON: {exc}")

    node_id = str(body.get("node_id") or "").strip()
    if not re.fullmatch(r"\d+", node_id):
        return web.Response(status=400, text="Invalid Director node id.")

    # Same key the executor uses: <cache_name>_<node_id> when 缓存文件夹名 is
    # filled, node id otherwise. Never trust the raw string as a path segment.
    from .segment_cache import resolve_segment_cache_key

    cache_key = resolve_segment_cache_key(body.get("cache_name"), node_id)
    if not cache_key:
        return web.Response(status=400, text="Invalid segment cache name.")

    timeline_data = body.get("timeline_data") or ""
    if isinstance(timeline_data, dict):
        timeline_data = json.dumps(timeline_data, ensure_ascii=False)
    try:
        from .external_groups import external_witness_from_timeline_data
        from .plan import build_director_plan
        from .segment_cache import inspect_first_pass_cache

        plan = build_director_plan(
            str(timeline_data),
            global_task_type=str(body.get("task_type") or ""),
            global_prompt=str(body.get("global_prompt") or ""),
            total_frames=int(body.get("total_frames") or 124),
            frame_rate=float(body.get("frame_rate") or 24.0),
            width=int(body.get("width") or 864),
            height=int(body.get("height") or 480),
            ref_max_size=int(body.get("ref_max_size") or 864),
        )
        plan.sample_seed = int(body.get("seed") or 0)
        plan.sample_cfg = float(body.get("cfg") or 1.0)
        plan.sample_steps = int(body.get("steps") or 25)
        plan.sample_sampler = str(body.get("sampler") or "")
        plan.sample_scheduler = str(body.get("scheduler") or "")
        plan.sample_sigmas_linked = bool(body.get("sigmas_linked"))
        plan.sample_shift_video = float(body.get("shift_video") or 12.0)
        plan.sample_shift_audio = float(body.get("shift_audio") or 3.0)
        from .selflift.pack import normalize_selflift_pack
        from .semantic_bridge import normalize_semantic_bridge_pack
        from .refine_pack import normalize_refine_pack

        plan.selflift = normalize_selflift_pack(body.get("selflift"))
        plan.semantic_bridge = normalize_semantic_bridge_pack(body.get("semantic_bridge"))
        plan.refine = normalize_refine_pack(
            body.get("refine"),
            base_width=int(getattr(plan, "width", 0) or 0),
            base_height=int(getattr(plan, "height", 0) or 0),
        )
        # Graph-wired i2v_groups / r2v_groups never reach this route as values,
        # so the panel ships a witness of that wiring inside timeline_data.
        witness = external_witness_from_timeline_data(timeline_data)
        if witness:
            plan.external_groups_witness = witness
        return web.json_response(
            inspect_first_pass_cache(cache_key, plan, external_groups=witness)
        )
    except Exception as exc:
        log.warning("MiniMax H3 Director first-pass cache inspection failed: %s", exc)
        return web.json_response(
            {"exists": False, "matches": False, "error": str(exc)},
            status=400,
        )


async def minimax_clear_segment_cache(request):
    """Delete first-pass (.pre.*) or final segment cache files."""
    try:
        body = await request.json()
    except Exception as exc:
        return web.Response(status=400, text=f"Invalid JSON: {exc}")

    node_id = str(body.get("node_id") or "").strip()
    if not re.fullmatch(r"\d+", node_id):
        return web.Response(status=400, text="Invalid Director node id.")

    kind = str(body.get("kind") or "final").strip().lower()
    if kind not in {"first_pass", "final", "all"}:
        return web.Response(status=400, text="kind must be first_pass, final or all.")

    try:
        from .segment_cache import clear_segment_cache, resolve_segment_cache_key

        cache_key = resolve_segment_cache_key(body.get("cache_name"), node_id)
        if not cache_key:
            return web.Response(status=400, text="Invalid segment cache name.")

        removed = clear_segment_cache(cache_key, kind=kind)
        return web.json_response({"removed": removed, "kind": kind})
    except Exception as exc:
        log.warning("MiniMax H3 Director clear segment cache failed: %s", exc)
        return web.Response(status=500, text=str(exc))


def _register_route(routes, method: str, path: str, handler) -> None:
    if hasattr(routes, "add_route"):
        routes.add_route(method, path, handler)
    elif method == "POST" and hasattr(routes, "post"):
        routes.post(path)(handler)
    elif method == "GET" and hasattr(routes, "get"):
        routes.get(path)(handler)
    else:
        raise AttributeError("Unsupported ComfyUI route table API")


def register_routes() -> bool:
    """Register MiniMax H3 Director HTTP routes on the ComfyUI PromptServer."""
    global _ROUTES_REGISTERED
    if _ROUTES_REGISTERED:
        return True

    server = PromptServer.instance
    if server is None:
        log.warning("MiniMax H3 Director: PromptServer not ready, HTTP routes not registered")
        return False

    routes = server.routes
    _register_route(routes, "POST", "/minimax/director/upload_chunk", minimax_upload_video_chunk)
    _register_route(
        routes,
        "POST",
        "/minimax/director/extract_reference_audio",
        minimax_extract_reference_audio,
    )
    _register_route(
        routes,
        "POST",
        "/minimax/director/prepare_reference_audio_chunk",
        minimax_prepare_reference_audio_chunk,
    )
    _register_route(routes, "POST", "/minimax/director/probe_video", minimax_probe_video)
    _register_route(routes, "GET", "/minimax/director/probe_video", minimax_probe_video)
    _register_route(routes, "GET", "/minimax/director/list_input_media", minimax_list_input_media)
    _register_route(routes, "GET", "/minimax/director/media_roots", minimax_media_roots)
    _register_route(routes, "GET", "/minimax/director/media_file", minimax_media_file)
    _register_route(routes, "POST", "/minimax/director/detect_shots", minimax_detect_shots)
    _register_route(
        routes,
        "POST",
        "/minimax/director/first_pass_cache_status",
        minimax_first_pass_cache_status,
    )
    _register_route(
        routes,
        "POST",
        "/minimax/director/clear_segment_cache",
        minimax_clear_segment_cache,
    )
    from .pack import minimax_download_pack, minimax_export_pack, minimax_import_pack

    _register_route(routes, "POST", "/minimax/director/export_pack", minimax_export_pack)
    _register_route(routes, "GET", "/minimax/director/download_pack", minimax_download_pack)
    _register_route(routes, "POST", "/minimax/director/import_pack", minimax_import_pack)
    from .lora_previews import minimax_lora_preview, minimax_lora_previews

    _register_route(routes, "GET", "/minimax/director/lora_previews", minimax_lora_previews)
    _register_route(routes, "GET", "/minimax/director/lora_preview", minimax_lora_preview)
    _ROUTES_REGISTERED = True
    log.info("MiniMax H3 Director HTTP routes registered")
    return True
