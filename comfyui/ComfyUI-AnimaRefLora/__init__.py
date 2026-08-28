"""ComfyUI nodes using AnimaRefLora's exact standalone inference path."""
from .bootstrap import bootstrap_paths

bootstrap_paths()

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
