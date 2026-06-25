"""ComfyUI-Krea2TexTEncoder — vision-aware text conditioning for the Krea2 / K2 model."""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

# Serves web/ as a frontend extension (auto-growing image inputs).
WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
