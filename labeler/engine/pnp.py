"""PnP solvers for pose estimation."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

try:
    import cv2
except ImportError as exc:  # pragma: no cover - dependency guard
    raise ImportError("OpenCV (cv2) is required for PnP operations") from exc

from .projection import undistort_points


class PnPError(RuntimeError):
    """Raised when pose estimation fails."""



def _explicit_arrays(session) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    ids = []
    pts3d = []
    pts2d = []
    for pid, obs in session.obs2d_explicit.items():
        if session.state.get(pid) == "OCCLUDED":
            continue
        ids.append(pid)
        pts3d.append(session.config.points_3d[pid])
        pts2d.append(obs)
    if not ids:
        raise PnPError("No explicit correspondences available")
    return np.asarray(ids), np.asarray(pts3d, dtype=np.float64), np.asarray(pts2d, dtype=np.float64)


def initial_pnp(session) -> Tuple[np.ndarray, np.ndarray]:
    ids, pts3d, pts2d = _explicit_arrays(session)
    if len(ids) < session.params.get("require_min_explicit_for_pose", 3):
        raise PnPError("Insufficient explicit points for pose estimation")

    reproj = float(session.params.get("ransac_reproj_px", 3.0))
    iters = int(session.params.get("ransac_iters", 1500))

    if len(ids) == 3:
        success, rvec, tvec = cv2.solvePnP(
            objectPoints=pts3d,
            imagePoints=pts2d,
            cameraMatrix=session.config.K_cam,
            distCoeffs=session.config.dist,
            flags=cv2.SOLVEPNP_SQPNP,
            useExtrinsicGuess=False,
        )
    else:
        success, rvec, tvec, _ = cv2.solvePnPRansac(
            objectPoints=pts3d,
            imagePoints=pts2d,
            cameraMatrix=session.config.K_cam,
            distCoeffs=session.config.dist,
            flags=cv2.SOLVEPNP_EPNP,
            reprojectionError=reproj,
            iterationsCount=iters,
        )
    if not success:
        raise PnPError("solvePnPRansac failed to find a pose")

    if hasattr(cv2, "solvePnPRefineLM"):
        rvec, tvec = cv2.solvePnPRefineLM(
            objectPoints=pts3d,
            imagePoints=pts2d,
            cameraMatrix=session.config.K_cam,
            distCoeffs=session.config.dist,
            rvec=rvec,
            tvec=tvec,
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, session.params.get("max_iter_LM", 20), 1e-6),
        )

    R, _ = cv2.Rodrigues(rvec)
    return R, tvec.reshape(3)


def iterative_pnp(session) -> Tuple[np.ndarray, np.ndarray]:
    ids, pts3d, pts2d = _explicit_arrays(session)
    if len(ids) < session.params.get("require_min_explicit_for_pose", 3):
        raise PnPError("Insufficient explicit points for pose refinement")

    if session.pose is None:
        return initial_pnp(session)

    R_init, t_init = session.pose
    rvec_init, _ = cv2.Rodrigues(R_init)
    success, rvec, tvec = cv2.solvePnP(
        objectPoints=pts3d,
        imagePoints=pts2d,
        cameraMatrix=session.config.K_cam,
        distCoeffs=session.config.dist,
        rvec=rvec_init,
        tvec=t_init.reshape(3, 1),
        useExtrinsicGuess=True,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        return initial_pnp(session)

    if hasattr(cv2, "solvePnPRefineLM"):
        rvec, tvec = cv2.solvePnPRefineLM(
            objectPoints=pts3d,
            imagePoints=pts2d,
            cameraMatrix=session.config.K_cam,
            distCoeffs=session.config.dist,
            rvec=rvec,
            tvec=tvec,
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, session.params.get("max_iter_LM", 20), 1e-6),
        )
    R, _ = cv2.Rodrigues(rvec)
    return R, tvec.reshape(3)
