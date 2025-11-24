import math
from typing import List, Tuple

import numpy as np


def build_parametric_points(preset: str) -> Tuple[np.ndarray, List[str]]:
    preset = preset.lower()
    if preset == "cube" or preset == "cube_3d":
        coords = [
            (-0.5, -0.5, -0.5),
            (0.5, -0.5, -0.5),
            (-0.5, 0.5, -0.5),
            (0.5, 0.5, -0.5),
            (-0.5, -0.5, 0.5),
            (0.5, -0.5, 0.5),
            (-0.5, 0.5, 0.5),
            (0.5, 0.5, 0.5),
        ]
        labels = [
            "back_bottom_left",
            "back_bottom_right",
            "back_top_left",
            "back_top_right",
            "front_bottom_left",
            "front_bottom_right",
            "front_top_left",
            "front_top_right",
        ]
    elif preset == "ring_spokes":
        angles = np.linspace(0, 2 * math.pi, 12, endpoint=False)
        coords = [(math.cos(a), math.sin(a), 0.0) for a in angles] + [(0.0, 0.0, 0.0)]
        labels = [f"rim_{i}" for i in range(len(angles))] + ["hub"]
    elif preset == "kite_3d":
        coords = [
            (0.0, 0.0, 0.0),
            (0.6, 0.0, 0.0),
            (0.2, 0.4, 0.0),
            (0.2, -0.4, 0.0),
            (0.2, 0.0, 0.2),
            (0.2, 0.0, -0.2),
        ]
        labels = ["origin", "tip_A", "wing_top", "wing_bottom", "mast_front", "mast_back"]
    elif preset == "random_cloud":
        rng = np.random.default_rng(0)
        coords = rng.uniform(-0.5, 0.5, size=(20, 3)).tolist()
        labels = [f"pt_{i}" for i in range(len(coords))]
    else:
        raise ValueError(f"Unknown preset: {preset}")
    return np.asarray(coords, dtype=float), labels


def look_at_rotation(camera_pos: np.ndarray, target: np.ndarray, roll_rad: float) -> np.ndarray:
    forward = target - camera_pos
    forward = forward / np.linalg.norm(forward)
    temp_up = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(forward, temp_up)) > 0.99:
        temp_up = np.array([0.0, 1.0, 0.0])
    right = np.cross(forward, temp_up)
    right = right / np.linalg.norm(right)
    up = np.cross(right, forward)
    R = np.stack([right, up, forward], axis=1)
    if abs(roll_rad) > 1e-8:
        c, s = math.cos(roll_rad), math.sin(roll_rad)
        roll_R = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
        R = R @ roll_R
    return R


def sample_on_sphere(rng: np.random.Generator, radius: float, az_rad: float, el_rad: float) -> np.ndarray:
    x = radius * math.cos(el_rad) * math.cos(az_rad)
    y = radius * math.cos(el_rad) * math.sin(az_rad)
    z = radius * math.sin(el_rad)
    return np.array([x, y, z], dtype=float)


def angular_distance_deg(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    dot = float(np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b) + 1e-9))
    dot = max(min(dot, 1.0), -1.0)
    return math.degrees(math.acos(dot))


def triangle_quality(bearings: np.ndarray) -> float:
    a, b, c = bearings
    ab = angular_distance_deg(a, b)
    bc = angular_distance_deg(b, c)
    ca = angular_distance_deg(c, a)
    return min(ab, bc, ca)


__all__ = [
    "build_parametric_points",
    "look_at_rotation",
    "sample_on_sphere",
    "angular_distance_deg",
    "triangle_quality",
]
