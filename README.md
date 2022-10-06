# Wildlife Re-Identification (Re-ID) Datasets

This package provides:
- overview of 30 publicly available wildlife re-identification datasets.
- utilities to mass download and convert them into a unified format.

## Overview

We show an example of downloading, extracting and processing the MacaqueFaces dataset.

```
from wildlife_datasets import datasets
from wildlife_datasets import utils

datasets.MacaqueFaces.download.get_data('data/MacaqueFaces')
dataset = datasets.MacaqueFaces('data/MacaqueFaces')
```

The class `dataset` contains the summary of the dataset. The content depends on the dataset. Each dataset contains the identity and paths to images. This particular dataset also contains information about the date taken and contrast. Other datasets store information about bounding boxes, segmentation masks, position from which the image was taken, keypoints or various other information such as axe or gender.

```
dataset.df
```

![](images/MacaqueFaces_DataFrame.png)

The dataset also contains basic metadata including information about the number of individuals, time span, licences or published year.

```
dataset.metadata
```

![](images/MacaqueFaces_Metadata.png)

This particular dataset already contains cropped images of faces. Other datasets may contain uncropped images with bounding boxes or even segmentation masks.

```
utils.analysis.plot_grid(dataset.df, 'data/MacaqueFaces')
```

![](images/MacaqueFaces_Grid.png)


## Installation

The installation is simple by
```
pip install wildlife-datasets
```

## Summary of datasets



| Dataset                |   Status    |  Method |             Comments           |
|------------------------|:-----------:|:-------:|-------------------------------:|
| AAU Zebrafish          | FINISHED    | AUTO    | Kaggle - CLI                   |
| Aerial Cattle          | FINISHED    | AUTO    |                                |
| ATRW                   | FINISHED    | AUTO    |                                |
| Beluga ID              | FINISHED    | AUTO    |                                |
| Bird Individual ID     | FINISHED    | MANUAL  | Manual G-Drive + Kaggle CLI    |
| Chimpanzee C-Tai       | FINISHED    | AUTO    |                                |
| Chimpanzee C-Zoo       | FINISHED    | AUTO    |                                |
| Cows 2021              | FINISHED    | AUTO    |                                |
| Drosophila             | FINISHED    | AUTO    | 90 GB 40 parts                 |
| Friesian Cattle 2015   | FINISHED    | AUTO    |                                |
| Friesian Cattle 2017   | FINISHED    | AUTO    |                                |
| Giraffe Zebra ID       | FINISHED    | AUTO    |                                |
| HappyWhale             | FINISHED    | MANUAL  | Kaggle - Need to agree terms   |
| HumpbackWhale          | FINISHED    | MANUAL  | Kaggle - Need to agree terms   |
| Hyena ID               | FINISHED    | AUTO    |                                |
| iPanda-50              | FINISHED    | AUTO    | G-Drive                        |
| Leopard ID             | FINISHED    | AUTO    |                                |
| Lion Data              | FINISHED    | AUTO    |                                |
| Macaque Faces          | FINISHED    | AUTO    |                                |
| NDD20                  | FINISHED    | AUTO    |                                |
| NOAA RightWhale        | FINISHED    | MANUAL  | Kaggle - Need to agree terms   |
| Nyala Data             | FINISHED    | AUTO    |                                |
| OpenCow 2020           | FINISHED    | AUTO    |                                |
| SealID                 | FINISHED    | MANUAL  | Download with single use token |
| SMALST                 | FINISHED    | AUTO    | Only linux: extract            |
| StripeSpotter          | FINISHED    | AUTO    | Only linux: extract            |
| Whale Shark ID         | FINISHED    | AUTO    |                                |
| WNI Giraffes           | FINISHED    | AUTO    | Only linux: download, 190 GB   |
| Zindi Turtle Recall    | FINISHED    | AUTO    |                                |

