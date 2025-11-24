import math
from typing import List, Tuple

import numpy as np

from .geometry import angular_distance_deg, triangle_quality


def synthesize_click_script(
    image_id: str,
    xy: np.ndarray,
    visible: np.ndarray,
    cfg: dict,
    rng: np.random.Generator,
) -> dict:
    events: List[dict] = []
    vis_indices = np.where(visible)[0]
    if vis_indices.size == 0:
        return {"image_id": image_id, "events": events}

    bearings = xy[vis_indices]
    bearings_centered = bearings - np.mean(bearings, axis=0)
    norms = np.linalg.norm(bearings_centered, axis=1) + 1e-9
    bearings_normed = bearings_centered / norms[:, None]

    anchor_indices = select_anchor_triplet(vis_indices, bearings_normed, rng, cfg.get("bootstrap_anchors", 3))
    for idx in anchor_indices:
        events.append({"op": "select_anchor", "id": int(idx)})
        noisy_click = add_noise(xy[idx], cfg.get("click_noise_sigma_px", 0.0), rng)
        events.append({"op": "click", "xy": noisy_click.tolist()})

    extra = int(cfg.get("extra_clicks", 0))
    if extra > 0:
        extra_ids = rng.choice(vis_indices, size=min(extra, vis_indices.size), replace=False)
        for idx in extra_ids:
            noisy_click = add_noise(xy[idx], cfg.get("click_noise_sigma_px", 0.0), rng)
            dist = float(np.linalg.norm(noisy_click - xy[idx]))
            if dist > cfg.get("snap_radius_px", 12.0):
                events.append({"op": "select_anchor", "id": int(idx)})
            events.append({"op": "click", "xy": noisy_click.tolist()})
    return {"image_id": image_id, "events": events}


def select_anchor_triplet(indices: np.ndarray, bearings_normed: np.ndarray, rng: np.random.Generator, count: int) -> List[int]:
    ids = [int(i) for i in indices.tolist()]
    if len(ids) <= count:
        return ids

    id_to_bearing = {pid: bearings_normed[i] for i, pid in enumerate(ids)}
    chosen: List[int] = []

    first = int(rng.choice(ids))
    chosen.append(first)
    remaining = [i for i in ids if i != first]
    if count == 1:
        return chosen

    second = int(max(remaining, key=lambda i: float(np.linalg.norm(id_to_bearing[i] - id_to_bearing[first]))))
    chosen.append(second)
    remaining = [i for i in remaining if i != second]

    while len(chosen) < count and remaining:
        best_idx = None
        best_score = -1.0
        for i in remaining:
            bearings = np.stack([id_to_bearing[c] for c in chosen[:2] + [i]])
            q = triangle_quality(bearings)
            if q > best_score:
                best_score = q
                best_idx = i
        chosen.append(int(best_idx))
        remaining = [i for i in remaining if i != best_idx]
    return chosen[:count]


def add_noise(pt: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    if sigma <= 0:
        return pt.copy()
    return pt + rng.normal(0.0, sigma, size=pt.shape)


__all__ = ["synthesize_click_script"]
