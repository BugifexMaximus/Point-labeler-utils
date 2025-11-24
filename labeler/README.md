# Labeler engine (prototype)

This package sketches the headless Python engine described in the labeling specification. It exposes pure functions for session management, pose solving, projection, and nearest-projection snapping, making it easy to plug into a UI layer such as PyQt5, DearPyGui, or a web client.

Key modules:
- `labeler.engine.session`: dataclasses for `ProjectConfig` and `ImageSession` plus session bootstrap helpers.
- `labeler.engine.events`: state-machine events (anchor selection, clicks, promotion, occlusion, undo) that keep RIGID and EXPLICIT layers synchronized.
- `labeler.engine.pnp`: Initial and iterative PnP solvers (RANSAC P3P/EPnP with LM refinement) that always operate on EXPLICIT correspondences only.
- `labeler.engine.nn2d`: nearest-projection search with snap-radius gating and tie-breaking for unset points.
- `labeler.engine.projection`: Brown–Conrady projection and undistortion helpers (via OpenCV).
- `labeler.engine.io`: import/export for project assets and per-image annotations.

The engine is UI-agnostic; build a thin client that calls the exported functions defined in `events.py` to realize the full anchor-first workflow described in the prompt.
