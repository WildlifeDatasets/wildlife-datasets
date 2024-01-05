# Wildlife Re-Identification (Re-ID) Datasets

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/WildlifeDatasets/wildlife-datasets/blob/main/LICENSE)
[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://wildlifedatasets.github.io/wildlife-datasets/)

The aim of the project is to provide comprehensive overview of datasets for wildlife individual re-identification and an easy-to-use package for developers of machine learning methods. The core functionality includes:

- overview of 31 publicly available wildlife re-identification datasets.
- utilities to mass download and convert them into a unified format.
- default splits for several machine learning tasks including the ability create additional splits.
- evaluation metrics for closed-set and open-set classification.

We provide [documentation](https://wildlifedatasets.github.io/wildlife-datasets/) and examples in [notebooks](https://github.com/WildlifeDatasets/wildlife-datasets/tree/main/notebooks). The package provides a natural synergy with [Wildlife tools](https://github.com/WildlifeDatasets/wildlife-tools), which provides out MegaDescriptor model and tools for training neural networks.

## Summary of datasets

An overview of the provided datasets is available in the [documentation](https://wildlifedatasets.github.io/wildlife-datasets/datasets/), while the more numerical summary is located in a [Jupyter notebook](notebooks/dataset_descriptions.ipynb). Due to its size, it may be necessary to view it via [nbviewer](https://nbviewer.org/github/WildlifeDatasets/wildlife-datasets/blob/main/notebooks/dataset_descriptions.ipynb).

We include basic characteristics such as publication years, number of images, number of individuals, dataset time span (difference between the last and first image taken) and additional information such as source, number of poses, inclusion of timestamps, whether the animals were captured in the wild and whether the dataset contain multiple species.

![](images/Datasets_Summary.png)


## Installation

The installation of the package is simple by
```
pip install wildlife-datasets
```


## Basic functionality

We show an example of downloading, extracting and processing the MacaqueFaces dataset.

```
from wildlife_datasets import analysis, datasets

datasets.MacaqueFaces.get_data('data/MacaqueFaces')
dataset = datasets.MacaqueFaces('data/MacaqueFaces')
```

The class `dataset` contains the summary of the dataset. The content depends on the dataset. Each dataset contains the identity and paths to images. This particular dataset also contains information about the date taken and contrast. Other datasets store information about bounding boxes, segmentation masks, position from which the image was taken, keypoints or various other information such as age or gender.

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
analysis.plot_grid(dataset.df, 'data/MacaqueFaces')
```

![](images/MacaqueFaces_Grid.png)

## Additional functionality

For additional functionality including mass loading, datasets splitting or evaluation metrics we refer to the [documentation](https://wildlifedatasets.github.io/wildlife-datasets/).

