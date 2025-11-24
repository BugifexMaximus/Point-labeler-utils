# Point-labeler-utils

Utilities for generating synthetic point-labeling datasets.

The `simulation/` folder contains a procedural renderer and dataset writer that follows the required dataset layout. See `simulation/README.md` for usage and configuration examples.

## Labeler engine

The `labeler/` directory contains a headless Python engine implementing the anchor-first 2D/3D point labeling workflow described in the accompanying prompt. It exposes pure functions for session management, projection, PnP pose solving, nearest-projection snapping, occlusion handling, and JSON export, ready to be wired into a thin UI layer.
