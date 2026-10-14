/**
 * Root-aware media URLs for the MiniMax H3 Director.
 *
 * The Director can reference media from several server-side "media roots":
 * ComfyUI's `input` and `output` directories plus any extra directories the user
 * configured in `media_roots.json`.
 *
 * Stock ComfyUI's `/view` endpoint only understands input/output/temp, so extra
 * roots are served through our own `/minimax/director/media_file` route. This
 * module is the single place that knows the difference; every other module
 * builds thumbnails/previews through it.
 */
import { api } from "../../scripts/api.js";

export const COMFY_VIEW_TYPES = new Set(["input", "output", "temp"]);
export const MEDIA_ROOTS_ENDPOINT = "/minimax/director/media_roots";
export const MEDIA_FILE_ENDPOINT = "/minimax/director/media_file";
export const DEFAULT_MEDIA_ROOT = "input";

const rootsState = {
    promise: null,
    roots: [],
    defaultId: DEFAULT_MEDIA_ROOT,
};

/** Normalise anything (string or entry object) to a media root id. */
export function mediaRootId(value) {
    if (value && typeof value === "object") {
        const raw = value.type ?? value.root ?? value.rootId ?? value.mediaRoot ?? "";
        return String(raw || DEFAULT_MEDIA_ROOT).trim() || DEFAULT_MEDIA_ROOT;
    }
    return String(value || DEFAULT_MEDIA_ROOT).trim() || DEFAULT_MEDIA_ROOT;
}

/** Normalise anything (string or entry object) to a forward-slash relative path. */
export function mediaRelPath(value) {
    if (value && typeof value === "object") {
        const raw = value.relPath
            ?? value.imageFile
            ?? value.videoFile
            ?? value.audioFile
            ?? value.previewImageFile
            ?? value.fileName
            ?? "";
        return String(raw || "").replace(/\\/g, "/").replace(/^\/+/, "");
    }
    return String(value || "").replace(/\\/g, "/").replace(/^\/+/, "");
}

/** URL that serves the given media (string path or entry object). */
export function mediaViewUrl(entry, fallbackType = DEFAULT_MEDIA_ROOT) {
    const rel = mediaRelPath(entry);
    if (!rel) return "";
    const rootId = entry && typeof entry === "object"
        ? mediaRootId(entry)
        : (String(fallbackType || DEFAULT_MEDIA_ROOT).trim() || DEFAULT_MEDIA_ROOT);

    if (!COMFY_VIEW_TYPES.has(rootId)) {
        const params = new URLSearchParams({ id: rootId, rel });
        return api.apiURL(`${MEDIA_FILE_ENDPOINT}?${params.toString()}`);
    }

    const slash = rel.lastIndexOf("/");
    const filename = slash >= 0 ? rel.slice(slash + 1) : rel;
    const subfolder = slash >= 0 ? rel.slice(0, slash) : "";
    const params = new URLSearchParams({ filename, type: rootId });
    if (subfolder) params.set("subfolder", subfolder);
    return api.apiURL(`/view?${params.toString()}`);
}

/**
 * Stable identity for a media entry: two files may share a relative path in
 * different roots (e.g. input/mutsuki/a.png and output/mutsuki/a.png), so the
 * root must be part of every dedupe/preselect key.
 */
export function mediaKey(entry) {
    const rel = mediaRelPath(entry);
    if (!rel) return "";
    return `${mediaRootId(entry)}:${rel}`;
}

/** Split a mediaKey back into `{ type, relPath }` (tolerates legacy bare paths). */
export function parseMediaKey(key, fallbackType = DEFAULT_MEDIA_ROOT) {
    const text = String(key || "");
    const at = text.indexOf(":");
    if (at > 0) {
        const type = text.slice(0, at);
        const relPath = text.slice(at + 1);
        if (relPath) return { type, relPath };
    }
    return { type: String(fallbackType || DEFAULT_MEDIA_ROOT), relPath: text };
}

/** Fetch the configured roots once per page (labels for the picker tabs). */
export function loadMediaRoots(force = false) {
    if (!force && rootsState.promise) return rootsState.promise;
    rootsState.promise = api.fetchApi(MEDIA_ROOTS_ENDPOINT)
        .then((resp) => (resp.ok ? resp.json() : null))
        .then((data) => {
            const roots = Array.isArray(data?.roots) ? data.roots : [];
            rootsState.roots = roots;
            rootsState.defaultId = String(data?.default || DEFAULT_MEDIA_ROOT);
            return roots;
        })
        .catch(() => {
            rootsState.promise = null; // allow a retry on next open
            return [];
        });
    return rootsState.promise;
}

export function listMediaRootsSync() {
    return rootsState.roots;
}
