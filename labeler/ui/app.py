"""Thin OpenCV + Open3D UI wiring the headless engine into an interactive tool."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Optional, Tuple

import cv2
import numpy as np
try:
    import open3d.visualization.gui as gui
    import open3d.visualization.rendering as rendering
except ImportError as exc:  # pragma: no cover - exercised only when Open3D missing
    gui = None
    rendering = None
    _OPEN3D_IMPORT_ERROR = exc

from ..engine import events
from ..engine.io import load_annotations, load_project, save_annotations
from ..engine.session import STATE_EXPLICIT, STATE_OCCLUDED, STATE_RIGID, ImageSession, start_session
from .overlays import render_overlay


_GUI_INITIALIZED = False
DEFAULT_PROJECT_ENV = "LABELER_DEFAULT_PROJECT"
SIM_DEFAULT_PROJECT = (
    Path(__file__).resolve().parents[2]
    / "simulation"
    / "synth_dataset"
    / "project"
)


def _ensure_gui_initialized() -> gui.Application:
    """Initialize the Open3D GUI singleton exactly once."""

    global _GUI_INITIALIZED
    if gui is None:
        raise ImportError(
            "Open3D is required for the 3D anchor pane; install open3d to enable the UI"
        ) from _OPEN3D_IMPORT_ERROR
    app = gui.Application.instance
    if not _GUI_INITIALIZED:
        app.initialize()
        _GUI_INITIALIZED = True
    return app


def resolve_default_paths(
    project_arg: Optional[str], image_arg: Optional[str]
) -> Tuple[Path, Path]:
    """Resolve project/image paths with simulation-friendly defaults.

    - If `project_arg` is provided, it wins.
    - Else, use `$LABELER_DEFAULT_PROJECT` if set; otherwise default to
      `simulation/synth_dataset/project` (the output location of the bundled
      simulation pipeline).
    - For the image, use the provided argument when present; otherwise pick the
      first PNG/JPG under `<project>/images`.
    """

    if project_arg:
        project = Path(project_arg)
    else:
        project = Path(os.environ.get(DEFAULT_PROJECT_ENV, SIM_DEFAULT_PROJECT))

    if not project.exists():
        raise FileNotFoundError(
            f"Project directory {project} not found. Generate one via the simulation "
            "pipeline (simulation/dataset.py) or point the labeler at an existing project."
        )

    if image_arg:
        image = Path(image_arg)
    else:
        images_dir = project / "images"
        candidates = sorted(images_dir.glob("*.png")) or sorted(images_dir.glob("*.jpg"))
        if not candidates:
            raise FileNotFoundError(
                f"No PNG/JPG images found under {images_dir}. Provide --image explicitly or "
                "generate a simulation run with images."
            )
        image = candidates[0]

    return project, image


class Anchor3DPane:
    """Lightweight Open3D scene for anchor selection and state feedback."""

    def __init__(self, session: ImageSession, on_select: Callable[[int], None], window_title: str = "Anchor (3D)"):
        self.session = session
        self.on_select = on_select
        self.window_title = window_title

        self.app = _ensure_gui_initialized()

        self.window = self.app.create_window(self.window_title, 640, 480)
        self.widget = gui.SceneWidget()
        self.widget.scene = rendering.Open3DScene(self.window.renderer)
        self.window.set_on_layout(self._on_layout)
        self.window.add_child(self.widget)

        self.material = rendering.MaterialRecord()
        self.material.shader = "defaultUnlit"
        self.material.point_size = 8.0

        self._points = session.config.points_3d
        self._pcd_name = "points"
        self._pick_radius_px = 18.0

        self._refresh_geometry()
        self.widget.set_on_mouse(self._on_mouse)

    def _on_layout(self, _):
        self.widget.frame = self.window.content_rect

    def _refresh_geometry(self) -> None:
        import open3d as o3d

        colors = []
        for idx, _ in enumerate(self._points):
            state = self.session.state.get(idx, STATE_RIGID)
            if state == STATE_OCCLUDED:
                colors.append([0.4, 0.4, 0.4])
            elif idx == self.session.active_anchor_id:
                colors.append([1.0, 0.3, 0.3])
            elif state == STATE_EXPLICIT:
                colors.append([0.0, 0.8, 1.0])
            else:
                colors.append([1.0, 0.85, 0.4])

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self._points)
        pcd.colors = o3d.utility.Vector3dVector(np.asarray(colors))

        self.widget.scene.remove_geometry(self._pcd_name)
        self.widget.scene.add_geometry(self._pcd_name, pcd, self.material)
        bbox = pcd.get_axis_aligned_bounding_box()
        self.widget.setup_camera(60.0, bbox, bbox.get_center())

    def _project_to_screen(self) -> Optional[np.ndarray]:
        if self.widget.frame.width == 0 or self.widget.frame.height == 0:
            return None
        cam = self.widget.scene.camera
        view = np.asarray(cam.get_view_matrix(), dtype=np.float64).reshape(4, 4).T
        proj = np.asarray(cam.get_projection_matrix(), dtype=np.float64).reshape(4, 4).T
        pts_h = np.concatenate([self._points, np.ones((len(self._points), 1))], axis=1)
        clip = (proj @ view @ pts_h.T).T
        clip[:, :3] /= clip[:, 3:4]
        # normalized device coords -> pixels
        x = (clip[:, 0] * 0.5 + 0.5) * float(self.widget.frame.width)
        y = (1.0 - (clip[:, 1] * 0.5 + 0.5)) * float(self.widget.frame.height)
        return np.stack([x, y], axis=1)

    def _on_mouse(self, event):
        if event.type == gui.MouseEvent.Type.BUTTON_DOWN and event.is_button_down(
            gui.MouseButton.LEFT
        ):
            screen_pts = self._project_to_screen()
            if screen_pts is None:
                return gui.Widget.EventCallbackResult.IGNORED
            click = np.asarray([event.x, event.y], dtype=np.float64)
            dists = np.linalg.norm(screen_pts - click[None, :], axis=1)
            idx = int(np.argmin(dists))
            if dists[idx] <= self._pick_radius_px:
                self.on_select(idx)
                self._refresh_geometry()
                return gui.Widget.EventCallbackResult.HANDLED
        return gui.Widget.EventCallbackResult.IGNORED

    def update_colors(self):
        self._refresh_geometry()

    def tick(self):
        self.app.run_one_tick()


class ImagePane:
    """OpenCV pane that renders overlay and forwards clicks to the engine."""

    def __init__(self, session: ImageSession, image: np.ndarray):
        self.session = session
        self.image = image
        self.status_text = ""
        self.window_name = "Labeler (2D)"
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self._on_mouse)

    def _on_mouse(self, event, x, y, *_):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        try:
            events.click_image(self.session, (float(x), float(y)))
            self.status_text = ""
        except ValueError as exc:  # snap gate failures, missing anchor, etc.
            self.status_text = str(exc)

    def render(self):
        canvas = render_overlay(self.session, self.image, self.status_text)
        cv2.imshow(self.window_name, canvas)


class LabelerUI:
    """Convenience wrapper that wires the headless engine into OpenCV/Open3D."""

    def __init__(self, project_path: str, image_path: str, annotations_path: Optional[str] = None):
        self.project_path = Path(project_path)
        self.image_path = Path(image_path)
        self.annotations_path = Path(annotations_path) if annotations_path else None

        self.config = load_project(str(self.project_path))
        self.image = cv2.imread(str(self.image_path))
        if self.image is None:
            raise FileNotFoundError(f"Could not read image from {image_path}")

        height, width = self.image.shape[:2]
        self.session = start_session(self.config, {"image_id": self.image_path.stem, "image_size": (height, width)})
        if self.annotations_path and self.annotations_path.exists():
            load_annotations(self.session, str(self.annotations_path))
            events.refresh_pose_and_rigid(self.session)

        self.image_pane = ImagePane(self.session, self.image)
        self.anchor_pane = Anchor3DPane(self.session, on_select=self._on_anchor_selected)

    def _on_anchor_selected(self, pid: int):
        events.select_anchor(self.session, pid)
        self.anchor_pane.update_colors()
        self.image_pane.status_text = f"Anchor set to {pid}"

    def _toggle_occlusion(self):
        if self.session.active_anchor_id is None:
            self.image_pane.status_text = "Pick an anchor before toggling occlusion"
            return
        pid = self.session.active_anchor_id
        currently = self.session.state.get(pid, STATE_RIGID) == STATE_OCCLUDED
        events.mark_occluded(self.session, pid, not currently)
        self.image_pane.status_text = f"{'Unhid' if currently else 'Hid'} point {pid}"
        self.anchor_pane.update_colors()

    def _unpromote_active(self):
        if self.session.active_anchor_id is None:
            self.image_pane.status_text = "No anchor selected to unpromote"
            return
        events.unpromote_point(self.session, self.session.active_anchor_id)
        self.anchor_pane.update_colors()

    def _handle_key(self, key: int) -> bool:
        if key == ord("q"):
            return False
        if key == ord("z"):
            events.undo(self.session)
            self.anchor_pane.update_colors()
            self.image_pane.status_text = "Undid last action"
        elif key == ord("o"):
            self._toggle_occlusion()
        elif key == ord("l"):
            self._unpromote_active()
        elif key == ord("s"):
            if self.annotations_path is None:
                self.image_pane.status_text = "No annotations path provided"
            else:
                save_annotations(self.session, str(self.annotations_path))
                self.image_pane.status_text = f"Saved {self.annotations_path}"
        elif key == ord("c"):
            events.clear_anchor(self.session)
            self.anchor_pane.update_colors()
            self.image_pane.status_text = "Cleared anchor"
        return True

    def run(self):
        running = True
        while running:
            self.anchor_pane.tick()
            self.image_pane.render()
            key = cv2.waitKey(10) & 0xFF
            if key != 255:
                running = self._handle_key(key)
        cv2.destroyAllWindows()
        gui.Application.instance.quit()


def launch_from_cli(args: Optional[list[str]] = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="OpenCV/Open3D labeler prototype")
    parser.add_argument(
        "project",
        nargs="?",
        default=None,
        help=(
            "Project directory; defaults to the simulation output folder if the "
            f"${DEFAULT_PROJECT_ENV} environment variable is set or to {SIM_DEFAULT_PROJECT}"
        ),
    )
    parser.add_argument(
        "image",
        nargs="?",
        default=None,
        help="Image to annotate; defaults to the first PNG/JPG under <project>/images",
    )
    parser.add_argument("--annotations", help="Optional annotations JSON path to load/save")
    parsed = parser.parse_args(args)

    project_path, image_path = resolve_default_paths(parsed.project, parsed.image)
    ui = LabelerUI(str(project_path), str(image_path), parsed.annotations)
    ui.run()


if __name__ == "__main__":
    launch_from_cli()
