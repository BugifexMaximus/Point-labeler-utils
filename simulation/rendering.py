import math
import random
from typing import Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def render_background(cfg: dict, H: int, W: int, rng: np.random.Generator) -> np.ndarray:
    mode = cfg.get("background", "solid")
    if mode == "solid":
        color = tuple(int(c) for c in rng.integers(0, 255, size=3))
        img = np.full((H, W, 3), color, dtype=np.uint8)
    elif mode == "gradient":
        top = rng.integers(0, 255, size=3)
        bottom = rng.integers(0, 255, size=3)
        grad = np.linspace(0, 1, H, dtype=np.float32)[:, None, None]
        img = (top * (1 - grad) + bottom * grad).astype(np.uint8)
        img = np.broadcast_to(img, (H, W, 3)).copy()
    elif mode == "checker":
        tile = rng.integers(20, 80)
        colors = rng.integers(0, 255, size=(2, 3))
        xs = np.arange(W) // tile
        ys = np.arange(H) // tile
        pattern = (xs[None, :] + ys[:, None]) % 2
        img = colors[pattern]
    else:
        # Procedural/perlin-like noise via smoothed random field
        grid = rng.random((H // 8 + 1, W // 8 + 1, 3))
        img = np.zeros((H, W, 3), dtype=np.float32)
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                gi = i / 8
                gj = j / 8
                i0, j0 = int(gi), int(gj)
                di, dj = gi - i0, gj - j0
                c00 = grid[i0, j0]
                c01 = grid[i0, j0 + 1]
                c10 = grid[i0 + 1, j0]
                c11 = grid[i0 + 1, j0 + 1]
                img[i, j] = (
                    c00 * (1 - di) * (1 - dj)
                    + c01 * (1 - di) * dj
                    + c10 * di * (1 - dj)
                    + c11 * di * dj
                )
        img = (img * 255).astype(np.uint8)
    return img


def apply_occluders(img: np.ndarray, cfg: dict, xy: np.ndarray, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil, "RGBA")
    H, W = img.shape[:2]
    mask = np.zeros((H, W), dtype=bool)
    if not cfg.get("enable", True):
        return np.asarray(img_pil), mask

    min_count, max_count = cfg.get("count", [0, 0])
    num_occ = int(rng.integers(min_count, max_count + 1))
    for _ in range(num_occ):
        color = tuple(int(c) for c in rng.integers(20, 235, size=3)) + (200,)
        size_min, size_max = cfg.get("size_px", [20, 120])
        w = int(rng.integers(size_min, size_max + 1))
        h = int(rng.integers(size_min, size_max + 1))
        if xy.size > 0 and rng.random() < cfg.get("hit_keypoint_prob", 0.0):
            choice_idx = int(rng.choice(len(xy)))
            cx, cy = xy[choice_idx]
        else:
            cx = float(rng.integers(0, W))
            cy = float(rng.integers(0, H))
        shape = cfg.get("shape", "ellipse")
        bbox = [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]
        if shape == "rect":
            draw.rectangle(bbox, fill=color)
        elif shape == "poly":
            pts = [
                (bbox[0], bbox[1]),
                (bbox[2], bbox[1]),
                (bbox[2], bbox[3]),
                (bbox[0], bbox[3]),
            ]
            draw.polygon(pts, fill=color)
        else:
            draw.ellipse(bbox, fill=color)
        cx_i, cy_i = int(round(cx)), int(round(cy))
        rad_x, rad_y = w / 2, h / 2
        y_grid, x_grid = np.ogrid[0:H, 0:W]
        inside = ((x_grid - cx) ** 2) / (rad_x ** 2 + 1e-6) + ((y_grid - cy) ** 2) / (rad_y ** 2 + 1e-6) <= 1
        mask = np.logical_or(mask, inside)

    img_arr = np.asarray(img_pil)
    return img_arr, mask


def apply_photometrics(img: np.ndarray, cfg: dict, rng: np.random.Generator) -> np.ndarray:
    img_pil = Image.fromarray(img)
    phot = cfg
    if phot.get("brightness_jitter", 0.0) > 0:
        delta = float(rng.uniform(-phot["brightness_jitter"], phot["brightness_jitter"]))
        img_pil = Image.fromarray(np.clip(np.asarray(img_pil) * (1 + delta), 0, 255).astype(np.uint8))
    if phot.get("contrast_jitter", 0.0) > 0:
        mean = np.mean(img_pil)
        delta = float(rng.uniform(-phot["contrast_jitter"], phot["contrast_jitter"]))
        img_pil = Image.fromarray(np.clip((np.asarray(img_pil) - mean) * (1 + delta) + mean, 0, 255).astype(np.uint8))
    if phot.get("add_noise", False):
        sigma = float(phot.get("noise_sigma_px", 0.0))
        noise = rng.normal(0.0, sigma, size=np.asarray(img_pil).shape)
        img_pil = Image.fromarray(np.clip(np.asarray(img_pil, dtype=float) + noise, 0, 255).astype(np.uint8))
    if phot.get("blur_kernel_odd", 0) in (3, 5):
        img_pil = img_pil.filter(ImageFilter.GaussianBlur(radius=(phot["blur_kernel_odd"] - 1) / 2))
    if phot.get("motion_blur_px", 0) >= 3:
        k = int(phot["motion_blur_px"])
        kernel = np.zeros((k, k))
        kernel[k // 2, :] = 1.0 / k
        img_arr = np.asarray(img_pil, dtype=float)
        img_pad = np.pad(img_arr, ((0, 0), (k // 2, k // 2), (0, 0)), mode="edge")
        convolved = np.zeros_like(img_arr)
        for i in range(k):
            convolved += kernel[k // 2, i] * img_pad[:, i:i + img_arr.shape[1], :]
        img_pil = Image.fromarray(np.clip(convolved, 0, 255).astype(np.uint8))
    return np.asarray(img_pil)


def draw_debug_keypoints(img: np.ndarray, xy: np.ndarray, visible: np.ndarray) -> np.ndarray:
    """Overlay small glyphs so projections are visually apparent in renders."""

    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    for pt, vis in zip(xy, visible):
        if not vis:
            continue
        x, y = pt
        r = 3
        draw.ellipse([x - r, y - r, x + r, y + r], fill=(255, 80, 80))
        draw.line([x - r * 2, y, x + r * 2, y], fill=(255, 255, 255), width=1)
        draw.line([x, y - r * 2, x, y + r * 2], fill=(255, 255, 255), width=1)
    return np.asarray(img_pil)


def write_image(path, img: np.ndarray) -> None:
    Image.fromarray(img).save(path)


__all__ = [
    "render_background",
    "apply_occluders",
    "apply_photometrics",
    "draw_debug_keypoints",
    "write_image",
]
