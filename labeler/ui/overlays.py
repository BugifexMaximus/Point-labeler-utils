"""OpenCV overlay helpers for rendering session layers."""

from __future__ import annotations

from typing import Dict, Tuple

import cv2
import numpy as np

from ..engine.events import get_layers
from ..engine.session import STATE_EXPLICIT, STATE_OCCLUDED


_COLOR_RIGID = (255, 220, 120)
_COLOR_EXPLICIT = (0, 200, 255)
_COLOR_OCCLUDED = (140, 140, 140)
_COLOR_RESIDUAL = (60, 60, 220)


def _draw_circles(canvas: np.ndarray, centers: Dict[int, np.ndarray], color: Tuple[int, int, int], filled: bool) -> None:
    for pt in centers.values():
        center = tuple(int(v) for v in pt)
        cv2.circle(canvas, center, 4, color, -1 if filled else 1, lineType=cv2.LINE_AA)


def _draw_residuals(canvas: np.ndarray, residuals: Dict[int, np.ndarray], rigid: Dict[int, np.ndarray]) -> None:
    for pid, vec in residuals.items():
        if pid not in rigid:
            continue
        start = tuple(int(v) for v in rigid[pid])
        end = tuple(int(v) for v in rigid[pid] + vec)
        cv2.arrowedLine(canvas, start, end, _COLOR_RESIDUAL, 1, tipLength=0.2)


def _draw_status(canvas: np.ndarray, session, status_text: str) -> None:
    """Render a tiny HUD with counts and last status message."""

    num_explicit = sum(1 for s in session.state.values() if s == STATE_EXPLICIT)
    num_occluded = sum(1 for s in session.state.values() if s == STATE_OCCLUDED)
    rmse = None
    if num_explicit and session.pose is not None:
        residuals = [
            np.linalg.norm(obs - session.rigid_proj2d[pid])
            for pid, obs in session.obs2d_explicit.items()
            if pid in session.rigid_proj2d
        ]
        if residuals:
            rmse = float(np.sqrt(np.mean(np.square(residuals))))

    lines = [
        f"EXPLICIT: {num_explicit}  OCCLUDED: {num_occluded}",
        f"RMSE px: {rmse:.2f}" if rmse is not None else "RMSE px: --",
    ]
    if status_text:
        lines.append(status_text)

    y = 24
    for text in lines:
        cv2.putText(canvas, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1, cv2.LINE_AA)
        y += 22


def render_overlay(session, image: np.ndarray, status_text: str = "") -> np.ndarray:
    """Draw RIGID/EXPLICIT layers and residuals onto an image copy."""

    canvas = image.copy()
    layers = get_layers(session)

    rigid = layers["rigid_proj2d"]
    explicit = layers["explicit_obs2d"]
    residuals = layers["residuals"]

    _draw_circles(canvas, rigid, _COLOR_RIGID, filled=False)
    _draw_circles(canvas, explicit, _COLOR_EXPLICIT, filled=True)
    _draw_residuals(canvas, residuals, rigid)
    _draw_status(canvas, session, status_text)
    return canvas
