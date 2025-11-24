import math
from typing import Tuple

import numpy as np


def project_points(
    P3D: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    img_size: Tuple[int, int],
):
    W, H = img_size
    X_c = (R @ P3D.T).T + t[None, :]
    in_front = X_c[:, 2] > 1e-6
    x_n = X_c[:, 0] / (X_c[:, 2] + 1e-9)
    y_n = X_c[:, 1] / (X_c[:, 2] + 1e-9)

    if dist.size in (4, 5):
        k1, k2, p1, p2 = dist[:4]
        k3 = dist[4] if dist.size == 5 else 0.0
        r2 = x_n ** 2 + y_n ** 2
        radial = 1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
        x_r = x_n * radial
        y_r = y_n * radial
        x_d = x_r + 2 * p1 * x_n * y_n + p2 * (r2 + 2 * x_n ** 2)
        y_d = y_r + p1 * (r2 + 2 * y_n ** 2) + 2 * p2 * x_n * y_n
    else:
        x_d, y_d = x_n, y_n

    u = K[0, 0] * x_d + K[0, 2]
    v = K[1, 1] * y_d + K[1, 2]
    xy = np.stack([u, v], axis=1)

    visible = np.logical_and.reduce(
        [in_front, u >= 0, u < W, v >= 0, v < H]
    )
    return xy, visible, in_front


def reprojection_error(xy_gt: np.ndarray, P3D: np.ndarray, R: np.ndarray, t: np.ndarray, K: np.ndarray, dist: np.ndarray, img_size: Tuple[int, int]) -> float:
    xy_proj, _, _ = project_points(P3D, R, t, K, dist, img_size)
    return float(np.sqrt(np.mean(np.sum((xy_gt - xy_proj) ** 2, axis=1))))


__all__ = ["project_points", "reprojection_error"]
