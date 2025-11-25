import pathlib
import sys

import numpy as np

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from labeler.engine.events import (
    click_image,
    clear_anchor,
    get_layers,
    get_pose,
    select_anchor,
    unpromote_point,
)
from labeler.engine.pnp import initial_pnp
from labeler.engine.projection import project_points
from labeler.engine.session import (
    STATE_EXPLICIT,
    STATE_RIGID,
    ProjectConfig,
    start_session,
)


def make_config():
    points_3d = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.2, 0.0, 0.1],
            [0.0, 0.2, 0.15],
            [0.1, 0.1, 0.25],
        ],
        dtype=np.float64,
    )
    K = np.array([[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]])
    dist = np.zeros(5)
    labels = [f"pt{i}" for i in range(len(points_3d))]
    return ProjectConfig(K_cam=K, dist=dist, points_3d=points_3d, labels=labels)


def test_initial_pnp_recovers_known_pose():
    cfg = make_config()
    session = start_session(cfg, {"image_id": "img", "image_size": (480, 640)})

    R_true = np.eye(3)
    t_true = np.array([0.0, 0.0, 5.0])
    projections = project_points(cfg.points_3d, cfg.K_cam, cfg.dist, R_true, t_true)

    for pid, uv in enumerate(projections):
        session.obs2d_explicit[pid] = uv
        session.state[pid] = STATE_EXPLICIT

    R_est, t_est = initial_pnp(session)

    reproj = project_points(cfg.points_3d, cfg.K_cam, cfg.dist, R_est, t_est)
    errors = np.linalg.norm(reproj - projections, axis=1)

    assert np.mean(errors) < 8.0
    assert t_est[2] > 0.1


def test_click_image_snaps_to_nearest_projection_when_pose_exists():
    cfg = make_config()
    session = start_session(cfg, {"image_id": "img", "image_size": (480, 640)})

    R_true = np.eye(3)
    t_true = np.array([0.0, 0.0, 5.0])
    projections = project_points(cfg.points_3d, cfg.K_cam, cfg.dist, R_true, t_true)

    # Bootstrap pose with anchors for three unique points.
    for pid in range(3):
        select_anchor(session, pid)
        click_image(session, tuple(projections[pid]))

    clear_anchor(session)
    assert get_pose(session) is not None

    target_click = tuple(projections[3] + np.array([2.0, -1.0]))
    click_image(session, target_click)

    assert session.state[3] == STATE_EXPLICIT
    assert np.allclose(session.obs2d_explicit[3], target_click)

    layers = get_layers(session)
    assert 3 in layers["rigid_proj2d"]
    assert np.linalg.norm(layers["rigid_proj2d"][3] - projections[3]) < 2.0


def test_unpromote_drops_pose_when_support_is_too_small():
    cfg = make_config()
    session = start_session(cfg, {"image_id": "img", "image_size": (480, 640)})

    R_true = np.eye(3)
    t_true = np.array([0.0, 0.0, 5.0])
    projections = project_points(cfg.points_3d, cfg.K_cam, cfg.dist, R_true, t_true)

    for pid in range(3):
        select_anchor(session, pid)
        click_image(session, tuple(projections[pid]))

    assert get_pose(session) is not None

    unpromote_point(session, 2)

    assert session.state[2] == STATE_RIGID
    assert get_pose(session) is None
    assert get_layers(session)["rigid_proj2d"] == {}
