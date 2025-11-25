"""Projection utilities for the labeling engine."""

from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np

try:
    import cv2
except ImportError as exc:  # pragma: no cover - dependency guard
    raise ImportError("OpenCV (cv2) is required for projection operations") from exc


def project_points(
    points_3d: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
) -> np.ndarray:
    """Project 3D points into the image plane.

    Returns an array of shape (N, 2) with pixel coordinates.
    """

    pts = np.asarray(points_3d, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("points_3d must be (N,3)")

    rvec, _ = cv2.Rodrigues(R)
    img_pts, _ = cv2.projectPoints(pts, rvec, np.asarray(t, dtype=np.float64), K, dist)
    return img_pts.reshape(-1, 2)


def undistort_points(points_px: Iterable[Tuple[float, float]], K: np.ndarray, dist: np.ndarray) -> np.ndarray:
    """Undistort pixel coordinates to normalized coordinates."""

    pts = np.asarray(list(points_px), dtype=np.float64).reshape(-1, 1, 2)
    undist = cv2.undistortPoints(pts, K, dist, P=K)
    return undist.reshape(-1, 2)
