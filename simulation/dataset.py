import json
import math
import pathlib
import subprocess
from typing import Dict, List, Tuple

import numpy as np
import yaml

from .click_sim import synthesize_click_script
from .config import SynthConfig
from .geometry import build_parametric_points, look_at_rotation, sample_on_sphere, angular_distance_deg
from .projection import project_points
from .rendering import apply_occluders, apply_photometrics, draw_debug_keypoints, render_background, write_image


def load_or_build_points(cfg: dict, rng: np.random.Generator) -> Tuple[np.ndarray, List[str]]:
    source = cfg.get("source", "parametric")
    scale = float(cfg.get("scale", 1.0))
    if source == "csv":
        path = cfg.get("csv_path")
        if path is None:
            raise ValueError("csv_path must be provided when source=csv")
        rows = np.genfromtxt(path, delimiter=",", skip_header=1, dtype=str)
        coords = rows[:, 1:4].astype(float) * scale
        labels = rows[:, 4].tolist()
    else:
        preset = cfg.get("preset", "kite_3D")
        coords, labels = build_parametric_points(preset)
        coords = coords * scale
    return coords, labels


def sample_intrinsics(cfg: dict, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    K_list = cfg.get("K")
    K = np.asarray(K_list, dtype=float).reshape(3, 3)
    dist = np.asarray(cfg.get("dist", []), dtype=float)
    if cfg.get("vary_intrinsics", False):
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        fx *= 1 + rng.uniform(-cfg.get("vary_fx_fy_pct", 0.0), cfg.get("vary_fx_fy_pct", 0.0))
        fy *= 1 + rng.uniform(-cfg.get("vary_fx_fy_pct", 0.0), cfg.get("vary_fx_fy_pct", 0.0))
        cx += rng.uniform(-cfg.get("vary_cx_cy_px", 0.0), cfg.get("vary_cx_cy_px", 0.0))
        cy += rng.uniform(-cfg.get("vary_cx_cy_px", 0.0), cfg.get("vary_cx_cy_px", 0.0))
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=float)
    return K, dist


def sample_pose(cfg: dict, accepted_dirs: List[np.ndarray], rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    ps = cfg
    attempts = 0
    while True:
        attempts += 1
        radius = rng.uniform(ps["radius_m"][0], ps["radius_m"][1])
        az = math.radians(rng.uniform(ps["az_deg"][0], ps["az_deg"][1]))
        el = math.radians(rng.uniform(ps["el_deg"][0], ps["el_deg"][1]))
        roll = math.radians(rng.uniform(ps["roll_deg"][0], ps["roll_deg"][1]))
        cam_pos = sample_on_sphere(rng, radius, az, el)
        target = np.asarray(ps.get("target", [0, 0, 0]), dtype=float)
        R = look_at_rotation(cam_pos, target, roll)
        t = -R @ cam_pos
        view_dir = (target - cam_pos) / (np.linalg.norm(target - cam_pos) + 1e-9)
        if accepted_dirs:
            angs = [angular_distance_deg(view_dir, v) for v in accepted_dirs]
            if min(angs) < ps.get("min_view_separation_deg", 0):
                continue
        return R, t


def append_annotation(records: List[dict], image_id: str, R: np.ndarray, t: np.ndarray, K: np.ndarray, dist: np.ndarray, xy: np.ndarray, visible: np.ndarray) -> None:
    entry = {
        "image_id": image_id,
        "pose": {
            "R": R.tolist(),
            "t": t.tolist(),
        },
        "keypoints": [
            {"id": int(i), "x": float(pt[0]) if vis else None, "y": float(pt[1]) if vis else None, "visible": int(bool(vis))}
            for i, (pt, vis) in enumerate(zip(xy, visible))
        ],
        "intrinsics": {
            "fx": float(K[0, 0]),
            "fy": float(K[1, 1]),
            "cx": float(K[0, 2]),
            "cy": float(K[1, 2]),
            "dist": dist.tolist(),
        },
    }
    records.append(entry)


def write_camera_yaml(path: pathlib.Path, K: np.ndarray, dist: np.ndarray) -> None:
    data = {"K": K.reshape(-1).tolist(), "dist": dist.tolist()}
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def write_points_csv(path: pathlib.Path, P3D: np.ndarray, labels: List[str]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("id,x,y,z,label\n")
        for i, (pt, lbl) in enumerate(zip(P3D, labels)):
            f.write(f"{i},{pt[0]:.3f},{pt[1]:.3f},{pt[2]:.3f},{lbl}\n")


def write_click_script(path: pathlib.Path, events: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(events, f, indent=2)


def git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=pathlib.Path(__file__).resolve().parent.parent).decode().strip()
    except Exception:
        return "unknown"


def generate_dataset(config_path: str, out_dir: str) -> None:
    out_root = pathlib.Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    cfg = SynthConfig.from_yaml(pathlib.Path(config_path))
    rng = np.random.default_rng(cfg["seed"])

    project_dir = out_root / "project"
    images_dir = project_dir / "images"
    clicks_dir = project_dir / "click_scripts"
    images_dir.mkdir(parents=True, exist_ok=True)
    clicks_dir.mkdir(parents=True, exist_ok=True)

    P3D, labels = load_or_build_points(cfg["object"], rng)
    write_points_csv(project_dir / "points_3d.csv", P3D, labels)

    K_ref, dist_ref = sample_intrinsics(cfg["camera"], rng)
    write_camera_yaml(project_dir / "camera.yaml", K_ref, dist_ref)

    ann_records: List[dict] = []
    accepted_dirs: List[np.ndarray] = []
    vis_counts: List[int] = []

    for idx in range(cfg["num_images"]):
        img_id = f"IMG_{idx + 1:06d}.png"
        # sample intrinsics per image (may vary)
        K, dist = sample_intrinsics(cfg["camera"], rng)
        dist_for_proj = dist if cfg["projection"].get("distortion_model", "brown_conrady") != "none" else np.array([])
        while True:
            R, t = sample_pose(cfg["pose_sampling"], accepted_dirs, rng)
            xy, visible, in_front = project_points(P3D, R, t, K, dist_for_proj, tuple(cfg["image_size"]))
            visibility_mask = visible if cfg["projection"].get("clip_to_image", True) else in_front
            if visibility_mask.sum() >= 4:
                # Store viewing direction for diversity enforcement
                accepted_dirs.append(-(R @ np.array([0, 0, 1.0])))
                break

        vis_counts.append(int(visibility_mask.sum()))

        bg = render_background(cfg["render"], cfg["image_size"][1], cfg["image_size"][0], rng)
        img, occ_mask = apply_occluders(bg, cfg["render"]["occluders"], xy, rng)
        if cfg["render"].get("keypoint_glyphs", "hidden") == "debug_small":
            label_mode = cfg["render"].get("keypoint_label_mode", "none")
            img = draw_debug_keypoints(img, xy, visibility_mask, tuple(labels), label_mode)

        phot_cfg = cfg["render"].get("photometric", {})
        img = apply_photometrics(img, phot_cfg, rng)

        if cfg["projection"].get("clip_to_image", True):
            visibility_mask = np.logical_and(visibility_mask, ~occ_mask[xy[:, 1].astype(int).clip(0, img.shape[0]-1), xy[:, 0].astype(int).clip(0, img.shape[1]-1)])

        write_image(images_dir / img_id, img)
        append_annotation(ann_records, img_id, R, t, K, dist_for_proj, xy, visibility_mask)

        if cfg["click_sim"].get("enable", False):
            events = synthesize_click_script(img_id, xy, visibility_mask, cfg["click_sim"], rng)
            write_click_script(clicks_dir / f"{img_id.replace('.png', '.clicks.json')}", events)

    annotations_path = project_dir / "annotations_gt.json"
    meta = {
        "seed": cfg["seed"],
        "git_hash": git_hash(),
        "num_images": cfg["num_images"],
        "image_size": cfg["image_size"],
    }
    with open(annotations_path, "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "images": ann_records}, f, indent=2)

    cfg.to_yaml(project_dir / "synth_config.yaml")

    splits_path = project_dir / "splits.json"
    n = cfg["num_images"]
    train_end = int(n * 0.8)
    val_end = int(n * 0.9)
    splits = {
        "train": list(range(0, train_end)),
        "val": list(range(train_end, val_end)),
        "test": list(range(val_end, n)),
    }
    with open(splits_path, "w", encoding="utf-8") as f:
        json.dump(splits, f, indent=2)

    readme_path = project_dir / "README.md"
    avg_vis = float(np.mean(vis_counts)) if vis_counts else 0.0
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write("# Synthetic Dataset\n")
        f.write(f"Images: {cfg['num_images']}\\n")
        f.write(f"Average visible keypoints: {avg_vis:.2f}\\n")
        f.write(f"Generator git hash: {meta['git_hash']}\\n")
        f.write("Visibility counts per image: " + ", ".join(map(str, vis_counts)) + "\n")


__all__ = [
    "generate_dataset",
    "load_or_build_points",
    "sample_intrinsics",
    "sample_pose",
    "project_points",
    "render_background",
    "apply_occluders",
    "apply_photometrics",
    "draw_debug_keypoints",
    "write_image",
    "write_camera_yaml",
    "write_points_csv",
    "append_annotation",
    "write_click_script",
]
