# NymeriaPlus

Official Python API for the **NymeriaPlus** dataset — large-scale egocentric multimodal recordings of human motion in the wild, with improved SMPL/MHR motion, dense object annotations, and additional modalities.

> Project page: https://www.projectaria.com/datasets/nymeria/ · Explorer: https://explorer.projectaria.com/nymeria

This package supports two workflows:

1. **Download** sequences from a JSON of signed URLs.
2. **Load + synchronize** a downloaded sequence into dense numpy arrays
   (one ``SynchronizedSequence`` per sequence, ready for training).

## Install With uv

The package requires Python 3.12 and is managed with
[uv](https://docs.astral.sh/uv/) by default. From the repository root:

```bash
# Linux/macOS: install uv if it is not already available
curl -LsSf https://astral.sh/uv/install.sh | sh
```

```bash
uv sync              # runtime deps, including projectaria-tools, torch, and pymomentum-cpu
uv sync --extra smpl # add SMPL body model support (smplx + scipy)
uv sync --extra all  # runtime deps plus all public optional features
uv sync --extra dev  # test/development dependencies
```

`pymomentum-cpu` depends on PyTorch's shared libraries at runtime. If importing
`pymomentum` fails with a missing `libtorch.so`, point `LD_LIBRARY_PATH` at the
uv environment's torch library directory before running scripts:

```bash
export LD_LIBRARY_PATH="$(pwd)/.venv/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
```

There is also a `gpu` extra for the CUDA pymomentum package on supported
platforms:

```bash
uv sync --extra gpu
```

## Install With Conda

For macOS MHR loading, use the Conda environment. It installs
`pymomentum-cpu>=0.1.110` from conda-forge with the matching PyTorch 2.10
runtime, which provides the native dependency stack missing from the
experimental PyPI macOS wheel.

```bash
conda env create -f environment.yml
conda activate nymeriaplus
python -m pip install -r requirements-conda-pip.txt
```

Run commands from the repository root:

```bash
python viewer.py \
    -i /path/to/nymeria_plus/<sequence> \
    --smpl-model-path /path/to/SMPL_n_lbs_300_207_0_v1.0.0.pkl
```

SMPL model files are not bundled with this package. To visualize the SMPL mesh,
download the SMPL model separately and pass its path with `--smpl-model-path`.

## Download

```bash
uv run python download.py -i <urls.json> -o <out_dir> [-k <substring>] [-y]
```

The `-k` flag filters sequences by a substring of the sequence name
(`<date>_<session>_<fake_name>_<act>_<uid>`).

## Viewer

Run the interactive viewer on a downloaded sequence:

```bash
uv run python viewer.py \
    -i /path/to/nymeria_plus/<sequence> \
    --smpl-model-path /path/to/SMPL_n_lbs_300_207_0_v1.0.0.pkl
```

Use `--target-fps` to change the synchronized playback rate. The viewer loads
the head RGB stream, semidense point cloud, Aria device trajectories, SMPL body,
and XSens skeleton when those files are present in the sequence directory.

## Load + synchronize

```python
from pathlib import Path
from nymeriaplus import NymeriaPlusDataLoader

loader = NymeriaPlusDataLoader(
    Path("/data/nymeria_plus/<sequence>"),
    smpl_model_path=Path("/path/to/SMPL_NEUTRAL.pkl"),
)
seq = loader.synchronize_data(target_fps=30.0)
seq.save_npz("./out/<sequence>.npz")
```

`SynchronizedSequence` holds dense numpy arrays (timestamps, SMPL params,
per-recording 4×4 trajectories) ready for training pipelines.

## Viewer Handoff

Current viewer command used during local development:

```bash
.venv/bin/python viewer.py \
    -i /home/summericequeen/data/nymeria_plus/20231130_s0_danielle_wu_act2_yugj1m \
    --smpl-model-path ~/data/SMPL/SMPL_n_lbs_300_207_0_v1.0.0.pkl
```

Useful validation command:

```bash
.venv/bin/python -m py_compile nymeriaplus/viewer.py
```

Current status:

- `LinesRenderer.draw()` requires a `viewport_size` argument. All viewer line-renderer calls, including `lines_xsens_r.draw(...)`, should pass `(w, h)`.
- Startup point-cloud loading should not immediately refilter the same points. The viewer state should initialize `_pcd_dirty` to `False`.
- XSens skeleton rendering from `body/xdata.npz` exists, but its transform into the Aria/world frame is still unresolved. Do not use `fbcode/surreal/nymeria/dataset/nymeria_synchronize.py` as the reference for that transform; it only covers the SMPL Y-up to Z-up correction path.

## Architecture

- `nymeriaplus.layout` — sequence on-disk layout, body model + recording enums, data-group definitions.
- `nymeriaplus.downloader` — `DownloadManager`, `DownloadLink`, `DownloadStatus`.
- `nymeriaplus.loaders` — one loader per modality (`RecordingLoader`, `SMPLBodyLoader`, MHR/XSens/Narration/Shaper/Boxy stubs).
- `nymeriaplus.data_loader` — `NymeriaPlusDataLoader` (multimodal entry point + `synchronize_data()`).
- `nymeriaplus.synchronized` — `SynchronizedSequence` dense data class (training-friendly).
- `download.py` (repo root) — Click-based CLI for batch downloading.

## License

CC BY-NC 4.0. See [LICENSE](./LICENSE).
