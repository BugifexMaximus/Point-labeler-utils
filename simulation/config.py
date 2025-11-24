import copy
import pathlib
from dataclasses import dataclass, field
from typing import Any, Dict

import yaml


DEFAULT_CONFIG: Dict[str, Any] = {
    "seed": 42,
    "num_images": 100,
    "image_size": [1280, 960],
    "camera": {
        "K": [1180.0, 0.0, 640.0, 0.0, 1182.0, 480.0, 0.0, 0.0, 1.0],
        "dist": [],
        "vary_intrinsics": False,
        "vary_fx_fy_pct": 0.05,
        "vary_cx_cy_px": 5,
    },
    "object": {
        "source": "parametric",
        "preset": "kite_3D",
        "csv_path": None,
        "scale": 1.0,
    },
    "pose_sampling": {
        "mode": "look_at",
        "target": [0.0, 0.0, 0.0],
        "radius_m": [1.2, 2.0],
        "az_deg": [-180, 180],
        "el_deg": [-30, 40],
        "roll_deg": [-10, 10],
        "min_view_separation_deg": 10,
    },
    "projection": {
        "distortion_model": "brown_conrady",
        "clip_to_image": True,
    },
    "render": {
        "background": "procedural",
        "keypoint_glyphs": "hidden",
        "occluders": {
            "enable": True,
            "count": [0, 4],
            "shape": "ellipse",
            "size_px": [20, 120],
            "hit_keypoint_prob": 0.3,
        },
        "photometric": {
            "add_noise": True,
            "noise_sigma_px": 0.8,
            "blur_kernel_odd": 0,
            "motion_blur_px": 0,
            "brightness_jitter": 0.1,
            "contrast_jitter": 0.1,
        },
    },
    "click_sim": {
        "enable": True,
        "click_noise_sigma_px": 1.0,
        "bootstrap_anchors": 3,
        "extra_clicks": 4,
        "snap_radius_px": 12.0,
        "misclick_rate": 0.0,
    },
}


@dataclass
class SynthConfig:
    data: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: pathlib.Path) -> "SynthConfig":
        user_cfg: Dict[str, Any]
        with open(path, "r", encoding="utf-8") as f:
            user_cfg = yaml.safe_load(f) or {}
        merged = deep_merge(copy.deepcopy(DEFAULT_CONFIG), user_cfg)
        return cls(merged)

    def to_yaml(self, path: pathlib.Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.data, f, sort_keys=False)

    def __getitem__(self, item: str) -> Any:
        return self.data[item]


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = deep_merge(base[key], value)
        else:
            base[key] = value
    return base


__all__ = ["SynthConfig", "DEFAULT_CONFIG", "deep_merge"]
