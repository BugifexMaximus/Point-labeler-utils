"""Event handlers implementing the labeling workflow."""

from __future__ import annotations

from copy import deepcopy
from typing import Dict, Tuple

import numpy as np

from .nn2d import find_nearest
from .pnp import PnPError, initial_pnp, iterative_pnp
from .projection import project_points
from .session import (
    ImageSession,
    STATE_EXPLICIT,
    STATE_OCCLUDED,
    STATE_RIGID,
)


def _update_rigid_layer(session: ImageSession) -> None:
    if session.pose is None:
        session.rigid_proj2d = {}
        return
    R, t = session.pose
    pts = project_points(session.config.points_3d, session.config.K_cam, session.config.dist, R, t)
    session.rigid_proj2d = {idx: pt for idx, pt in enumerate(pts) if session.state.get(idx) != STATE_OCCLUDED}


def _recompute_pose(session: ImageSession, force_initial: bool = False) -> None:
    min_explicit = session.params.get("require_min_explicit_for_pose", 3)
    num_explicit = sum(1 for pid, state in session.state.items() if state == STATE_EXPLICIT)
    if num_explicit < min_explicit:
        session.pose = None
        _update_rigid_layer(session)
        return

    solver = initial_pnp if force_initial or session.pose is None else iterative_pnp
    session.pose = solver(session)
    _update_rigid_layer(session)


def select_anchor(session: ImageSession, point_id: int) -> None:
    session.active_anchor_id = int(point_id)


def clear_anchor(session: ImageSession) -> None:
    session.active_anchor_id = None


def click_image(session: ImageSession, xy: Tuple[float, float]) -> None:
    """Handle a click in the 2D pane."""

    session.history.append(session.snapshot())

    min_explicit = session.params.get("require_min_explicit_for_pose", 3)
    num_explicit = sum(1 for pid, state in session.state.items() if state == STATE_EXPLICIT)
    pose_ready = session.pose is not None and num_explicit >= min_explicit

    if not pose_ready:
        if session.active_anchor_id is None:
            session.history.pop()
            raise ValueError("Select an anchor before placing early explicit observations")
        pid = session.active_anchor_id
        session.obs2d_explicit[pid] = np.asarray(xy, dtype=np.float64)
        session.state[pid] = STATE_EXPLICIT
        _recompute_pose(session, force_initial=True)
        return

    # pose exists; use anchor if explicitly set otherwise nearest rigid projection
    if session.active_anchor_id is not None:
        pid = session.active_anchor_id
    else:
        pid, dist = find_nearest(session, xy)
        if pid is None:
            session.history.pop()
            raise ValueError("No rigid projection within snap radius; pick an anchor instead")
    session.obs2d_explicit[pid] = np.asarray(xy, dtype=np.float64)
    session.state[pid] = STATE_EXPLICIT
    _recompute_pose(session)


def promote_point(session: ImageSession, point_id: int, xy: Tuple[float, float] | None = None) -> None:
    session.history.append(session.snapshot())
    pid = int(point_id)
    if xy is not None:
        session.obs2d_explicit[pid] = np.asarray(xy, dtype=np.float64)
    elif pid not in session.obs2d_explicit and pid in session.rigid_proj2d:
        session.obs2d_explicit[pid] = np.asarray(session.rigid_proj2d[pid], dtype=np.float64)
    elif pid not in session.obs2d_explicit:
        session.history.pop()
        raise ValueError("No observation provided for promotion")
    session.state[pid] = STATE_EXPLICIT
    _recompute_pose(session)


def unpromote_point(session: ImageSession, point_id: int) -> None:
    session.history.append(session.snapshot())
    pid = int(point_id)
    session.obs2d_explicit.pop(pid, None)
    if session.state.get(pid) != STATE_OCCLUDED:
        session.state[pid] = STATE_RIGID
    _recompute_pose(session)


def mark_occluded(session: ImageSession, point_id: int, flag: bool = True) -> None:
    session.history.append(session.snapshot())
    pid = int(point_id)
    if flag:
        session.state[pid] = STATE_OCCLUDED
        session.obs2d_explicit.pop(pid, None)
    else:
        session.state[pid] = STATE_RIGID
    _recompute_pose(session)


def set_params(session: ImageSession, **kwargs) -> None:
    session.history.append(session.snapshot())
    session.params.update(kwargs)
    _recompute_pose(session)


def undo(session: ImageSession) -> None:
    if not session.history:
        return
    snap = session.history.pop()
    session.restore(snap)


def get_pose(session: ImageSession):
    return deepcopy(session.pose)


def get_layers(session: ImageSession) -> Dict:
    residuals = {}
    if session.pose is not None:
        for pid, obs in session.obs2d_explicit.items():
            if pid in session.rigid_proj2d:
                residuals[pid] = np.asarray(obs) - np.asarray(session.rigid_proj2d[pid])
    return {
        "rigid_proj2d": deepcopy(session.rigid_proj2d),
        "explicit_obs2d": deepcopy(session.obs2d_explicit),
        "residuals": residuals,
        "candidates": deepcopy(session.candidates2d),
    }


def get_states(session: ImageSession) -> Dict[int, str]:
    return deepcopy(session.state)


def refresh_pose_and_rigid(session: ImageSession) -> None:
    """Public helper to recompute pose and rigid projections after loading annotations."""

    _recompute_pose(session, force_initial=True)
