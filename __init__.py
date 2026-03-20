from pathlib import Path
import warnings

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS


def _warn_if_legacy_ui_api_used() -> None:
    """Detect stale frontend files that still import deprecated /scripts/ui.js."""
    base = Path(__file__).resolve().parent
    for js_path in base.rglob("*.js"):
        try:
            content = js_path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if "/scripts/ui.js" in content:
            warnings.warn(
                f"Deprecated legacy API import found in {js_path}. "
                "Please remove/update this file to the new frontend API.",
                RuntimeWarning,
                stacklevel=2,
            )


_warn_if_legacy_ui_api_used()

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
