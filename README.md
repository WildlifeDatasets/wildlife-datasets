# Wildlife Re-Identification (Re-ID) Datasets

This package provides:
- overview of 30 publicly available [wildlife re-identification datasets](notebooks/dataset_descriptions.ipynb).
- utilities to mass download and convert them into a unified format.


## Installation

The installation is simple by
```
pip install wildlife-datasets
```


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


## Summary of datasets



| Dataset                | Method  |             Comments           |
|------------------------|:-------:|-------------------------------:|
| [AAU Zebrafish](https://www.kaggle.com/datasets/aalborguniversity/aau-zebrafish-reid)          | AUTO    | Kaggle - CLI                   |
| [Aerial Cattle](https://data.bris.ac.uk/data/dataset/3owflku95bxsx24643cybxu3qh)          | AUTO    |                                |
| [ATRW](https://lila.science/datasets/atrw)                   | AUTO    |                                |
| [Beluga ID](https://lila.science/datasets/beluga-id-2022/)              | AUTO    |                                |
| [Bird Individual ID](https://github.com/AndreCFerreira/Bird_individualID)     | MANUAL  | Manual G-Drive + Kaggle CLI    |
| [Chimpanzee C-Tai](https://github.com/cvjena/chimpanzee_faces)       | AUTO    |                                |
| [Chimpanzee C-Zoo](https://github.com/cvjena/chimpanzee_faces)       | AUTO    |                                |
| [Cows 2021](https://data.bris.ac.uk/data/dataset/4vnrca7qw1642qlwxjadp87h7)              | AUTO    |                                |
| [Drosophila](https://github.com/j-schneider/fly_eye)             | AUTO    |                                |
| [Friesian Cattle 2015](https://data.bris.ac.uk/data/dataset/wurzq71kfm561ljahbwjhx9n3)   | AUTO    |                                |
| [Friesian Cattle 2017](https://data.bris.ac.uk/data/dataset/2yizcfbkuv4352pzc32n54371r)   | AUTO    |                                |
| [Giraffe Zebra ID](https://lila.science/datasets/great-zebra-giraffe-id)       | AUTO    |                                |
| [HappyWhale](https://www.kaggle.com/competitions/happy-whale-and-dolphin)             | MANUAL  | Kaggle - Need to agree terms   |
| [HumpbackWhale](https://www.kaggle.com/competitions/humpback-whale-identification)          | MANUAL  | Kaggle - Need to agree terms   |
| [Hyena ID](https://lila.science/datasets/hyena-id-2022/)               | AUTO    |                                |
| [iPanda-50](https://github.com/iPandaDateset/iPanda-50)              | AUTO    |                                |
| [Leopard ID](https://lila.science/datasets/leopard-id-2022/)             | AUTO    |                                |
| [Lion Data](https://github.com/tvanzyl/wildlife_reidentification)              | AUTO    |                                |
| [Macaque Faces](https://github.com/clwitham/MacaqueFaces)          | AUTO    |                                |
| [NDD20](https://doi.org/10.25405/data.ncl.c.4982342)                  | AUTO    |                                |
| [NOAA RightWhale](https://www.kaggle.com/c/noaa-right-whale-recognition)        | MANUAL  | Kaggle - Need to agree terms   |
| [Nyala Data](https://github.com/tvanzyl/wildlife_reidentification)             | AUTO    |                                |
| [OpenCows 2020](https://data.bris.ac.uk/data/dataset/10m32xl88x2b61zlkkgz3fml17)           | AUTO    |                                |
| [SealID](https://doi.org/10.23729/0f4a3296-3b10-40c8-9ad3-0cf00a5a4a53)                 | MANUAL  | Download with single use token |
| SeaTurtleID                 | AUTO    |                                |
| [SMALST](https://github.com/silviazuffi/smalst)                 | AUTO    | Only linux: extract            |
| [StripeSpotter](https://code.google.com/archive/p/stripespotter/downloads)          | AUTO    | Only linux: extract            |
| [Whale Shark ID](https://lila.science/datasets/whale-shark-id)         | AUTO    |                                |
| [WNI Giraffes](https://lila.science/datasets/wni-giraffes)           | AUTO    | Only linux: download          |
| [Zindi Turtle Recall](https://zindi.africa/competitions/turtle-recall-conservation-challenge)    | AUTO    |                                |

