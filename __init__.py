from pathlib import Path
import warnings

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS


LEGACY_API_IMPORTS = [
    "/scripts/ui.js",
    "/extensions/core/widgetInputs.js",
    "/scripts/ui/components/buttonGroup.js",
    "/scripts/ui/components/button.js",
]


def _find_legacy_js_files(base: Path):
    matches = []
    for js_path in base.rglob("*.js"):
        try:
            content = js_path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        hit = [marker for marker in LEGACY_API_IMPORTS if marker in content]
        if hit:
            matches.append((js_path, hit))
    return matches


def _quarantine_legacy_js_files() -> None:
    """
    If stale frontend JS exists in this custom-node folder and imports deprecated
    legacy APIs, rename those files so ComfyUI no longer loads them.
    """
    base = Path(__file__).resolve().parent
    legacy_files = _find_legacy_js_files(base)
    if not legacy_files:
        return

    for js_path, hits in legacy_files:
        quarantined = js_path.with_suffix(js_path.suffix + ".legacy.disabled")
        if quarantined.exists():
            # already quarantined in previous run
            continue
        try:
            js_path.rename(quarantined)
            warnings.warn(
                "Quarantined stale frontend file using deprecated ComfyUI APIs: "
                f"{js_path} -> {quarantined}. Hits={hits}",
                RuntimeWarning,
                stacklevel=2,
            )
        except OSError as e:
            warnings.warn(
                "Detected deprecated frontend API usage but failed to quarantine file: "
                f"{js_path}. Hits={hits}. Error={e}",
                RuntimeWarning,
                stacklevel=2,
            )


_quarantine_legacy_js_files()

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
