"""Session state and helpers for the labeling engine."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .params import DEFAULT_PARAMS

StateType = str


@dataclass
class ProjectConfig:
    """Project-level configuration and 3D geometry."""

    K_cam: np.ndarray
    dist: np.ndarray
    points_3d: np.ndarray
    labels: List[str]
    graph_edges: Optional[np.ndarray] = None
    symmetry: Optional[dict] = None


@dataclass
class ImageSession:
    """State for annotating a single image."""

    image_id: str
    image_size: Tuple[int, int]
    config: ProjectConfig
    pose: Optional[Tuple[np.ndarray, np.ndarray]] = None
    obs2d_explicit: Dict[int, np.ndarray] = field(default_factory=dict)
    state: Dict[int, StateType] = field(default_factory=dict)
    rigid_proj2d: Dict[int, np.ndarray] = field(default_factory=dict)
    candidates2d: Dict[int, Tuple[float, float, float]] = field(default_factory=dict)
    active_anchor_id: Optional[int] = None
    params: Dict = field(default_factory=lambda: deepcopy(DEFAULT_PARAMS))
    history: List[dict] = field(default_factory=list)

    def snapshot(self) -> dict:
        """Return a lightweight snapshot for undo."""

        return {
            "pose": deepcopy(self.pose),
            "obs2d_explicit": deepcopy(self.obs2d_explicit),
            "state": deepcopy(self.state),
            "rigid_proj2d": deepcopy(self.rigid_proj2d),
            "active_anchor_id": self.active_anchor_id,
        }

    def restore(self, snap: dict) -> None:
        self.pose = snap["pose"]
        self.obs2d_explicit = snap["obs2d_explicit"]
        self.state = snap["state"]
        self.rigid_proj2d = snap["rigid_proj2d"]
        self.active_anchor_id = snap["active_anchor_id"]


STATE_EXPLICIT = "EXPLICIT"
STATE_RIGID = "RIGID"
STATE_OCCLUDED = "OCCLUDED"


def start_session(cfg: ProjectConfig, image_meta: dict) -> ImageSession:
    """Initialize a fresh image annotation session."""

    session = ImageSession(
        image_id=image_meta.get("image_id", ""),
        image_size=tuple(image_meta["image_size"]),
        config=cfg,
    )

    for idx in range(len(cfg.points_3d)):
        session.state[idx] = STATE_RIGID
    for occluded in image_meta.get("occluded", []):
        session.state[int(occluded)] = STATE_OCCLUDED
    return session
