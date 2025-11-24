# Simulation pipeline

This folder contains a lightweight renderer that generates datasets matching the requested structure. The main entry point is `generate_dataset(config_path, out_dir)` inside `simulation/dataset.py`.

## Usage

Install dependencies from `requirements.txt` (NumPy, Pillow, and PyYAML) before running the generator.

```
python -c "from simulation.dataset import generate_dataset; generate_dataset('simulation/synth_config.example.yaml', 'synth_dataset')"
```

Outputs will be placed in `synth_dataset/project` following the prescribed layout (images, annotations, camera intrinsics, click scripts, and config snapshots).
By default, small debug glyphs are rendered at each projected keypoint to make the object visible in the PNGs; set `render.keypoint_glyphs` to `hidden` if you prefer clean backgrounds. You can also annotate glyphs with text by setting `render.keypoint_label_mode` to `id`, `label`, or `id_label` (defaults to `none`).

### Quick verification

After installing dependencies, you can sanity-check the pipeline by running the example configuration above. It will produce a small dataset in `synth_dataset/`. Remove the folder afterward if you do not want to keep the generated assets:

```
rm -rf synth_dataset
```

## Components
- `config.py`: default configuration and loader that merges user overrides.
- `dataset.py`: orchestrator and serialization helpers.
- `geometry.py`: parametric point generators and pose math.
- `projection.py`: Brown–Conrady projection utilities.
- `rendering.py`: procedural backgrounds, occluders, and photometric effects.
- `click_sim.py`: anchor-first click script synthesis with optional noisy nearest-projection steps.

Adjust `synth_config.example.yaml` to exercise the pre-baked scenarios described in the specification (distortion, occlusion, blur, etc.).

## Working without internet access

If the execution environment blocks outbound network access, you can still enable the required third-party libraries by preparing wheels or source archives ahead of time:

1. **Collect wheels elsewhere**: On a machine with internet access, run `pip download -r simulation/requirements.txt -d wheelhouse/` to fetch compatible wheels for your Python version and platform.
2. **Transfer artifacts**: Copy the `wheelhouse/` directory into this repository (or any accessible path in the offline environment).
3. **Install offline**: Install the dependencies with `pip install --no-index --find-links wheelhouse/ -r simulation/requirements.txt`.

If you do have controlled proxy access instead of full internet, set the `HTTPS_PROXY` environment variable (for example, `export HTTPS_PROXY=http://user:pass@proxy:port`) before running `pip install -r simulation/requirements.txt`.
