"""Headless labeling engine entry points."""

from .events import (
    clear_anchor,
    click_image,
    get_layers,
    get_pose,
    get_states,
    mark_occluded,
    promote_point,
    select_anchor,
    set_params,
    undo,
    unpromote_point,
)
from .io import load_annotations, load_project, save_annotations
from .params import DEFAULT_PARAMS
from .pnp import PnPError
from .session import (
    ImageSession,
    ProjectConfig,
    STATE_EXPLICIT,
    STATE_OCCLUDED,
    STATE_RIGID,
    start_session,
)

__all__ = [
    "clear_anchor",
    "click_image",
    "get_layers",
    "get_pose",
    "get_states",
    "load_annotations",
    "load_project",
    "mark_occluded",
    "promote_point",
    "save_annotations",
    "select_anchor",
    "set_params",
    "undo",
    "unpromote_point",
    "DEFAULT_PARAMS",
    "PnPError",
    "ImageSession",
    "ProjectConfig",
    "STATE_EXPLICIT",
    "STATE_OCCLUDED",
    "STATE_RIGID",
    "start_session",
]
