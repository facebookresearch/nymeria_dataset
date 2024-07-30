# Nymeria Dataset

[[Project Page]](https://www.projectaria.com/datasets/nymeria/) [[Paper]](https://arxiv.org/abs/2406.09905) [[Web Browser]](https://explorer.projectaria.com/?v=%22Nymeria%22)

Nymeria is the world's largest dataset of human motion in the wild, capturing diverse people engaging in diverse activities across diverse locations. It is first of its kind to record body motion using multiple egocentric multimodal devices, all accurately synchronized and localized in one single metric 3D world. Nymeria also holds the distinction of the world's largest dataset of motion-language descriptions, featuring hierarchical in-context narration. The dataset is designed to accelerate research in egocentric human motion understanding and present exciting new challenges beyond body tracking, motion synthesis and action recognition. It aims to advance contextualized computing and pave the way for future AR/VR technology. This repository hosts the official API for downloading and loading the dataset.

<p align="center">
  <img src=".github/teaser1.gif" width="49%" alt="Nymeria dataset teaser with 100 random samples" />
  <img src=".github/teaser2.gif" width="49%" alt="Nymeria dataset highlight statistics" />
</p>

## Getting Started

### Prerequest

### Download the dataset
Review the dataset [LICENSE](./LICENSE) to ensure your application is covered before proceeding further. Upon agreeing with the license, visit [Nymeria dataset website](https://www.projectaria.com/datasets/nymeria/), scroll to the bottom of the page, and sign up with your email by **Access the Dataset**. The enrollment will abide you to the license, and take you to the download page. Click `DATA (Nymeria_download_urls.json)` to download the JSON file.

**About the JSON.** Nymeria dataset has more than 1200 sequences, and each sequence contains data/annotations recorded by multiple devices. Altogether the dataset is approximately 70TB. To easy data access, we chunk the dataset into sequences and sequences into data groups. A data group is a fixed collection of files, which must be downloaded together via one url. The JSON file is essentially a collection of all urls. To understand the predefined data groups, please refer the definitions specified by `GroupDefs` in [definition.py](https://github.com/facebookresearch/nymeria_dataset/blob/main/nymeria/definitions.py#L29-L83).

The JSON file contains all the information to access data. For convinience, we provide [download.py](./download.py) as an example script which parses the JSON file and downloads data into formatted directories.

```
# Activate environment
conda activate nymeria

# Go to your local copy of the code repository
cd nymeria_dataset

python download.py -i <Nymeria_download_urls.json> -o <dataset_outpath>
```
The download script will require your confirmation to proceed downloading. Under `<dataset_outpath>`, we produce a `download_summary.json` file to summarize the download status. For applications only require a subset of data, we implement 2 filters to configure download.

* Filter by sequence name. We name the sequences by the following convention `<date>_<session_id>_<fake_id>_<act_id>_<uid>`. This allow sequence selection by partial key matching. **TIP** you can use complex filters to select sequences from [data browser](https://explorer.projectaria.com/?v=%22Nymeria%22).
```
python download.py -i <Nymeria_download_urls.json> -o <dataset_outpath> -k <partial_key>

# E.g., get all sequences collected in June
python download.py -i <Nymeria_download_urls.json> -o <dataset_outpath> -k 202306

# E.g., get all sequences from participant with fake_id = james_johnson
python download.py -i <Nymeria_download_urls.json> -o <dataset_outpath> -k james_johnson

# E.g., get a particular sequence with uid egucf6
python download.py -i <Nymeria_download_urls.json> -o <dataset_outpath> -k egucf6
```

* Filter by data groups. Open [download.py](https://github.com/facebookresearch/nymeria_dataset/blob/main/download.py#L9-L29) and change the function `get_groups()`.



### Visualize the data

## License

Nymeria dataset and code is released by Meta under the Creative Commons Attribution-NonCommercial 4.0 International License
([CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/legalcode)).
Data and code may not be used for commercial purposes. For more information, please refer to the [LICENSE](./LICENSE) file included in this repository.

### Attribution

When using the dataset and code, please attribute it as follows:

```
@inproceedings{ma24eccv,
      title={Nymeria: A Massive Collection of Multimodal Egocentric Daily Motion in the Wild},
      author={Lingni Ma and Yuting Ye and Fangzhou Hong and Vladimir Guzov and Yifeng Jiang and Rowan Postyeni and Luis Pesqueira and Alexander Gamino and Vijay Baiyya and Hyo Jin Kim and Kevin Bailey and David Soriano Fosas and C. Karen Liu and Ziwei Liu and Jakob Engel and Renzo De Nardi and Richard Newcombe},
      booktitle={the 18th European Conference on Computer Vision (ECCV)},
      year={2024},
      url={https://arxiv.org/abs/2406.09905},
}
```

### Contribute

We welcome contributions! Go to [CONTRIBUTING](.github/CONTRIBUTING.md) and our [CODE OF CONDUCT](.github/CODE_OF_CONDUCT.md) for how to contribute.
