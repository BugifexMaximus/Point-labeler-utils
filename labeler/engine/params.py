"""Default parameters for the labeling engine."""

from __future__ import annotations

DEFAULT_PARAMS = {
    "snap_radius_px": 12.0,
    "ransac_iters": 1500,
    "ransac_reproj_px": 3.0,
    "robust_loss": {"type": "cauchy", "c": 2.0},
    "max_iter_LM": 20,
    "prefer_unset_on_tie": True,
    "require_min_explicit_for_pose": 3,
    "show_occluded": False,
}
