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

## OpenCV/Open3D UI prototype

The `labeler.ui` package wires the headless engine into a minimal desktop UI built with OpenCV (2D pane) and Open3D (3D anchor pane).

```
python -m labeler.ui.app /path/to/project /path/to/image.jpg --annotations /tmp/out.json
```

If you omit the positional arguments, the CLI will try to launch against the
latest simulation output (defaulting to `simulation/synth_dataset/project` or
`$LABELER_DEFAULT_PROJECT` when set) and pick the first PNG/JPG under
`<project>/images` for quick smoke testing.

Hotkeys:
- Click the 3D pane to set the active anchor (red highlight).
- Left-click the 2D pane to place/adjust EXPLICIT observations per the engine workflow.
- `o`: toggle OCCLUDED for the active anchor.
- `l`: unpromote the active anchor back to RIGID.
- `z`: undo last action.
- `c`: clear the anchor selection.
- `s`: save annotations to the provided path.
- `q`: quit.
