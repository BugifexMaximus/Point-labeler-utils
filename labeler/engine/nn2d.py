"""Nearest-neighbor helpers for rigid projections."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .session import STATE_OCCLUDED


def find_nearest(session, xy: Tuple[float, float]) -> Tuple[Optional[int], float]:
    """Return the closest point id and distance in pixels.

    Applies the snap radius gate stored on the session.
    """

    if not session.rigid_proj2d:
        return None, float("inf")

    coords = []
    ids = []
    for pid, pt in session.rigid_proj2d.items():
        if session.state.get(pid) == STATE_OCCLUDED:
            continue
        coords.append(pt)
        ids.append(pid)
    if not ids:
        return None, float("inf")

    pts = np.asarray(coords, dtype=np.float64)
    query = np.asarray(xy, dtype=np.float64)
    dists = np.linalg.norm(pts - query, axis=1)
    idx = int(np.argmin(dists))
    dist = float(dists[idx])
    if dist > float(session.params.get("snap_radius_px", 12.0)):
        return None, dist

    nearest_id = ids[idx]
    if session.params.get("prefer_unset_on_tie", True) and dist <= float(session.params.get("snap_radius_px", 12.0)):
        # Check for alternative unset candidate within tolerance
        unset_ids = [pid for pid in ids if pid not in session.obs2d_explicit]
        if unset_ids:
            unset_coords = np.asarray([session.rigid_proj2d[pid] for pid in unset_ids], dtype=np.float64)
            unset_dists = np.linalg.norm(unset_coords - query, axis=1)
            best_idx = int(np.argmin(unset_dists))
            if unset_dists[best_idx] <= dist + 1e-6:
                nearest_id = unset_ids[best_idx]
                dist = float(unset_dists[best_idx])
    return nearest_id, dist
