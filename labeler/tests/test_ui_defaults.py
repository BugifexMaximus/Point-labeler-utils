import tempfile
from pathlib import Path

import cv2
import numpy as np

from labeler.ui.app import DEFAULT_PROJECT_ENV, resolve_default_paths


def _write_dummy_project(root: Path) -> tuple[Path, Path]:
    project = root / "project"
    images = project / "images"
    project.mkdir(parents=True)
    images.mkdir()
    (project / "annotations").mkdir()

    (project / "camera.yaml").write_text(
        "K: [800, 0, 320, 0, 800, 240, 0, 0, 1]\ndist: []\n"
    )
    (project / "points_3d.csv").write_text("id,x,y,z,label\n0,0,0,0,pt0\n")

    img_path = images / "000000.png"
    cv2.imwrite(str(img_path), np.zeros((8, 8, 3), dtype=np.uint8))

    return project, img_path


def test_resolve_defaults_prefers_env_and_first_image(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        project, img = _write_dummy_project(Path(tmp))
        monkeypatch.setenv(DEFAULT_PROJECT_ENV, str(project))

        resolved_project, resolved_img = resolve_default_paths(None, None)

        assert resolved_project == project
        assert resolved_img == img


def test_resolve_defaults_respects_cli_overrides(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        project_a, _ = _write_dummy_project(Path(tmp) / "a")
        project_b, img_b = _write_dummy_project(Path(tmp) / "b")
        monkeypatch.delenv(DEFAULT_PROJECT_ENV, raising=False)

        resolved_project, resolved_img = resolve_default_paths(str(project_a), str(img_b))

        assert resolved_project == project_a
        assert resolved_img == img_b
