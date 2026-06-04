# Nymeria and NymeriaPlus Dataset

[[Nymeria Data Explorer]](https://explorer.projectaria.com/nymeria)
[[Nymeria Paper]](https://arxiv.org/abs/2406.09905)
[[NymeriaPlus Data Explorer]](https://explorer.projectaria.com/nymeria_plus)
[[NymeriaPlus Paper]](https://arxiv.org/abs/2603.18496)
[[Bibtex]](#attribution)

**Nymeria** is the world's largest dataset of human motion in the wild, capturing
diverse people performing diverse activities across diverse locations. It is
first of a kind to record body motion using multiple egocentric multimodal
devices, all accurately synchronized and localized in one metric 3D world.
Nymeria is also the world's largest motion dataset with natural language
descriptions. The dataset is designed to accelerate research in egocentric human
motion understanding and presents exciting challenges to advance contextualized
computing and future AR/VR technology.

<p align="center">
  <img src=".github/teaser1.gif" width="49%" alt="Nymeria dataset teaser with 100 random samples" />
  <img src=".github/teaser2.gif" width="49%" alt="Nymeria dataset highlight statistics" />
</p>

**NymeriaPlus** is an updated version of the original Nymeria dataset with
additional annnotations and data. NymeriaPlus features: (1) optimized human
motion in both [MHR](https://github.com/facebookresearch/MHR) and
[SMPL](https://smpl.is.tue.mpg.de/) formats; (2) dense 3D and 2D bounding box
annotations for indoor objects and structured elements; (3) instance-level 3D
object reconstruction based on [ShapeR](https://github.com/facebookresearch/ShapeR)
and human rating; and (4) additional modalities such as basemap recordings,
wristband videos, headset audio and etc.

<p align="center">
  <img src=".github/nymeria_plus_teaser.jpg" width="98%" alt="NymeriaPlus teaser showing additional annotations and modalities" />
</p>
<br>

This repository hosts the API for downloading and visualizing the dataset.
The `main` branch best supports NymeriaPlus dataset. For Nymeria dataset, switch to the
`nymeria_dataset_legacy` branch, if `main` does not work.

## Getting Started

### Installation

Clone the repository:

```
git clone git@github.com:facebookresearch/nymeria_dataset.git
cd nymeria_dataset
```

#### Option A: install with uv

This route works well on Linux and partially on MacOS if your workflow does not
require loading MHR human motion. The installation requires Python 3.12 and is
managed with [uv](https://docs.astral.sh/uv/) by default. From the repository
root:

```bash
# Linux/MacOS: install uv if it is not already available
curl -LsSf https://astral.sh/uv/install.sh | sh
```

```bash
uv sync              # runtime deps, including projectaria-tools, torch, and pymomentum-cpu
uv sync --extra gpu  # use CUDA-supported pymomentum-gpu on supported platforms
uv sync --extra smpl # add SMPL body model support
```

`pymomentum-cpu` depends on PyTorch's shared libraries at runtime. If importing
`pymomentum` fails with a missing `libtorch.so`, point `LD_LIBRARY_PATH` at the
uv environment's torch library directory before running scripts:

```bash
export LD_LIBRARY_PATH="$(pwd)/.venv/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
```

#### Option B: install with conda

If your workflow requires loading MHR human motion on MacOS, you need to use
Conda to install `pymomentum-cpu>=0.1.110` from conda-forge with the matching
PyTorch 2.10 runtime, which provides the native dependency stack missing from
the experimental PyPI macOS wheel.

```bash
conda env create -f environment.yml
conda activate nymeriaplus
python -m pip install -r requirements-conda-pip.txt
```

## License

This code is released by Meta under the Creative Commons
Attribution-NonCommercial 4.0 International License
([CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/legalcode)). Data
and code may not be used for commercial purposes. For more information, please
refer to the [LICENSE](./LICENSE) file included in this repository.

Use of the datasets is governed by their own terms, included at the repository
root: [NYMERIA_DATASET_LICENSE](./NYMERIA_DATASET_LICENSE) for the Nymeria
dataset and [NYMERIAPLUS_DATASET_LICENSE](./NYMERIAPLUS_DATASET_LICENSE) for the
NymeriaPlus dataset.

### Attribution

When using the Nymeria dataset and code, please attribute it as follows:

```bibtex
@inproceedings{nymeria24,
      title={Nymeria: A Massive Collection of Multimodal Egocentric Daily Motion in the Wild},
      author={Lingni Ma and Yuting Ye and Fangzhou Hong and Vladimir Guzov and Yifeng Jiang and Rowan Postyeni and Luis Pesqueira and Alexander Gamino and Vijay Baiyya and Hyo Jin Kim and Kevin Bailey and David Soriano Fosas and C. Karen Liu and Ziwei Liu and Jakob Engel and Renzo De Nardi and Richard Newcombe},
      booktitle={the 18th European Conference on Computer Vision (ECCV)},
      year={2024},
      url={https://arxiv.org/abs/2406.09905},
}
```

When using the NymeriaPlus dataset, please also attribute the following:

```bibtex
@misc{nymeriaplus26,
      title={NymeriaPlus: Enriching Nymeria Dataset with Additional Annotations and Data},
      author={Daniel DeTone and Federica Bogo and Eric-Tuan Le and Duncan Frost and Julian Straub and Yawar Siddiqui and Yuting Ye and Jakob Engel and Richard Newcombe and Lingni Ma},
      year={2026},
      eprint={2603.18496},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2603.18496},
}
```

### Contribute

We welcome contributions! Go to [CONTRIBUTING](.github/CONTRIBUTING.md) and our
[CODE OF CONDUCT](.github/CODE_OF_CONDUCT.md) for how to contribute.
