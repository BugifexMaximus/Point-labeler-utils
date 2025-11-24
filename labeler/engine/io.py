"""I/O helpers for project configs and annotations."""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from typing import Dict

import numpy as np
import yaml

from .session import ImageSession, ProjectConfig, STATE_EXPLICIT, STATE_OCCLUDED, STATE_RIGID


def load_project(config_path: str) -> ProjectConfig:
    """Load project assets from a directory containing camera.yaml and points_3d.csv."""

    if os.path.isdir(config_path):
        camera_path = os.path.join(config_path, "camera.yaml")
        points_path = os.path.join(config_path, "points_3d.csv")
        labels_path = os.path.join(config_path, "labels.txt")
    else:
        camera_path = config_path
        base = os.path.dirname(config_path)
        points_path = os.path.join(base, "points_3d.csv")
        labels_path = os.path.join(base, "labels.txt")

    with open(camera_path, "r", encoding="utf-8") as f:
        cam_data = yaml.safe_load(f)
    K = np.asarray(cam_data.get("K"), dtype=np.float64).reshape(3, 3)
    dist = np.asarray(cam_data.get("dist", []), dtype=np.float64).reshape(-1)

    points = []
    labels = []
    with open(points_path, "r", encoding="utf-8") as f:
        header = f.readline()
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 4:
                continue
            _, x, y, z, *rest = parts
            points.append([float(x), float(y), float(z)])
            labels.append(rest[0] if rest else "")
    if os.path.exists(labels_path):
        with open(labels_path, "r", encoding="utf-8") as f:
            labels = [line.strip() for line in f.readlines()]

    return ProjectConfig(
        K_cam=K,
        dist=dist,
        points_3d=np.asarray(points, dtype=np.float64),
        labels=labels,
    )


def save_annotations(session: ImageSession, path_json: str) -> None:
    """Serialize the session state to JSON."""

    output = {
        "image_id": session.image_id,
        "pose": None,
        "points": [],
        "qa": {},
    }
    if session.pose is not None:
        R, t = session.pose
        output["pose"] = {"R": R.reshape(-1).tolist(), "t": t.tolist()}

    for idx in range(len(session.config.points_3d)):
        state = session.state.get(idx, STATE_RIGID)
        point_entry: Dict = {
            "id": idx,
            "state": state,
        }
        if state == STATE_EXPLICIT and idx in session.obs2d_explicit:
            point_entry["obs"] = session.obs2d_explicit[idx].tolist()
        if idx in session.rigid_proj2d:
            point_entry["rigid"] = session.rigid_proj2d[idx].tolist()
            if state == STATE_EXPLICIT:
                err = np.linalg.norm(session.obs2d_explicit[idx] - session.rigid_proj2d[idx])
                point_entry["err_px"] = float(err)
        output["points"].append(point_entry)

    explicit_errs = [p.get("err_px", 0.0) for p in output["points"] if p.get("state") == STATE_EXPLICIT]
    output["qa"] = {
        "rmse_px": float(np.sqrt(np.mean(np.square(explicit_errs)))) if explicit_errs else None,
        "num_explicit": int(sum(1 for p in output["points"] if p.get("state") == STATE_EXPLICIT)),
    }

    with open(path_json, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)


def load_annotations(session: ImageSession, path_json: str) -> None:
    with open(path_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    session.obs2d_explicit.clear()
    session.state = {idx: STATE_RIGID for idx in range(len(session.config.points_3d))}
    session.pose = None

    pose_data = data.get("pose")
    if pose_data:
        R = np.asarray(pose_data.get("R"), dtype=np.float64).reshape(3, 3)
        t = np.asarray(pose_data.get("t"), dtype=np.float64).reshape(3)
        session.pose = (R, t)

    for entry in data.get("points", []):
        pid = int(entry["id"])
        state = entry.get("state", STATE_RIGID)
        session.state[pid] = state
        if state == STATE_EXPLICIT and "obs" in entry:
            session.obs2d_explicit[pid] = np.asarray(entry["obs"], dtype=np.float64)
    # rigid projections will be recomputed by the caller after loading
