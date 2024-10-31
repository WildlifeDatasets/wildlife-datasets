<p align="center">
  <a href="https://github.com/WildlifeDatasets/wildlife-datasets/issues"><img src="https://img.shields.io/github/issues/WildlifeDatasets/wildlife-datasets" alt="GitHub issues"></a>
  <a href="https://github.com/WildlifeDatasets/wildlife-datasets/pulls"><img src="https://img.shields.io/github/issues-pr/WildlifeDatasets/wildlife-datasets" alt="GitHub pull requests"></a>
  <a href="https://github.com/WildlifeDatasets/wildlife-datasets/graphs/contributors"><img src="https://img.shields.io/github/contributors/WildlifeDatasets/wildlife-datasets" alt="GitHub contributors"></a>
  <a href="https://github.com/WildlifeDatasets/wildlife-datasets/network/members"><img src="https://img.shields.io/github/forks/WildlifeDatasets/wildlife-datasets" alt="GitHub forks"></a>
  <a href="https://github.com/WildlifeDatasets/wildlife-datasets/stargazers"><img src="https://img.shields.io/github/stars/WildlifeDatasets/wildlife-datasets" alt="GitHub stars"></a>
  <a href="https://github.com/WildlifeDatasets/wildlife-datasets/watchers"><img src="https://img.shields.io/github/watchers/WildlifeDatasets/wildlife-datasets" alt="GitHub watchers"></a>
  <a href="https://github.com/WildlifeDatasets/wildlife-datasets/blob/main/LICENSE"><img src="https://img.shields.io/github/license/WildlifeDatasets/wildlife-datasets" alt="License"></a>
</p>

<p align="center">
<img src="https://github.com/WildlifeDatasets/wildlife-datasets/raw/main/docs/resources/datasets-logo.png" alt="Wildlife datasets" width="300">
</p>

<div align="center">
  <p align="center">Pipeline for wildlife re-identification including dataset zoo, training tools and trained models. Usage includes classifying new images in labelled databases and clustering individuals in unlabelled databases.</p>
  <a href="https://wildlifedatasets.github.io/wildlife-datasets/">Documentation</a>
  ·
  <a href="https://github.com/WildlifeDatasets/wildlife-datasets/issues/new?assignees=aerodynamic-sauce-pan&labels=bug&projects=&template=bug_report.md&title=%5BBUG%5D">Report Bug</a>
  ·
  <a href="https://github.com/WildlifeDatasets/wildlife-datasets/issues/new?assignees=aerodynamic-sauce-pan&labels=enhancement&projects=&template=enhancement.md&title=%5BEnhancement%5D">Request Feature</a>
  ·
  <a href="mailto:wilddatasets@gmail.com">:mailbox_with_mail:Email</a>
</div>

</br>

| <a href="https://github.com/WildlifeDatasets/wildlife-datasets"><img src="https://github.com/WildlifeDatasets/wildlife-datasets/raw/main/docs/resources/datasets-logo.png" alt="Wildlife datasets" width="200"></a>  | <a href="https://huggingface.co/BVRA/MegaDescriptor-L-384"><img src="https://github.com/WildlifeDatasets/wildlife-datasets/raw/main/docs/resources/megadescriptor-logo.png" alt="MegaDescriptor" width="200"></a> | <a href="https://github.com/WildlifeDatasets/wildlife-tools"><img src="https://github.com/WildlifeDatasets/wildlife-datasets/raw/main/docs/resources/tools-logo.png" alt="Wildlife tools" width="200"></a> |
|:--------------:|:-----------:|:------------:|
| Datasets for identification of individual animals | Trained model for individual re&#x2011;identification  | Tools for training re&#x2011;identification models |

</br>

## Wildlife Re-Identification (Re-ID) Datasets

The aim of the project is to provide comprehensive overview of datasets for wildlife individual re-identification and an easy-to-use package for developers of machine learning methods. The core functionality includes:

- overview of 42 publicly available wildlife re-identification datasets.
- utilities to mass download and convert them into a unified format and fix some wrong labels.
- default splits for several machine learning tasks including the ability create additional splits.

An introductory example is provided in a [Jupyter notebook](notebooks/introduction.ipynb). The package provides a natural synergy with [Wildlife tools](https://github.com/WildlifeDatasets/wildlife-tools), which provides our [MegaDescriptor](https://huggingface.co/BVRA/MegaDescriptor-L-384) model and tools for training neural networks. 

## Changelog

[31/10/2024] Added AmvrakikosTurtles, ReunionTurtles, SouthernProvinceTurtles, ZakynthosTurtles (sea turtles), ELPephants (elephants) and Chicks4FreeID (chickens).  
[13/06/2024] Added WildlifeReID-10k (unification of multiple datasets).  
[09/05/2024] Added CatIndividualImages (cats), CowDataset (cows) and DogFaceNet (dogs).  
[28/02/2024] Added MPDD (dogs), PolarBearVidID (polar bears) and SeaStarReID2023 (sea stars).  
[04/01/2024] Received **Best paper award** at WACV 2024.

## Summary of datasets

An overview of the provided datasets is available in the [documentation](https://wildlifedatasets.github.io/wildlife-datasets/datasets/), while the more numerical summary is located in a [Jupyter notebook](notebooks/dataset_descriptions.ipynb). Due to its size, it may be necessary to view it via [nbviewer](https://nbviewer.org/github/WildlifeDatasets/wildlife-datasets/blob/main/notebooks/dataset_descriptions.ipynb).

We include basic characteristics such as publication years, number of images, number of individuals, dataset time span (difference between the last and first image taken) and additional information such as source, number of poses, inclusion of timestamps, whether the animals were captured in the wild and whether the dataset contain multiple species.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://github.com/WildlifeDatasets/wildlife-datasets/raw/main/docs/resources/Datasets_Summary_inverted.png">
  <source media="(prefers-color-scheme: light)" srcset="https://github.com/WildlifeDatasets/wildlife-datasets/raw/main/docs/resources/Datasets_Summary.png">
  <img alt="Dataset summary" src="https://github.com/WildlifeDatasets/wildlife-datasets/raw/main/docs/resources/Datasets_Summary.png">
</picture>


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

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://github.com/WildlifeDatasets/wildlife-datasets/raw/main/docs/resources/MacaqueFaces_DataFrame_inverted.png">
  <source media="(prefers-color-scheme: light)" srcset="https://github.com/WildlifeDatasets/wildlife-datasets/raw/main/docs/resources/MacaqueFaces_DataFrame.png">
  <img alt="Overview of the MacaqueFaces dataset" src="https://github.com/WildlifeDatasets/wildlife-datasets/raw/main/docs/resources/MacaqueFaces_DataFrame.png">
</picture>

The dataset also contains basic metadata including information about the number of individuals, time span, licences or published year.

```
dataset.summary
```

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://github.com/WildlifeDatasets/wildlife-datasets/raw/main/docs/resources/MacaqueFaces_Metadata_inverted.png">
  <source media="(prefers-color-scheme: light)" srcset="https://github.com/WildlifeDatasets/wildlife-datasets/raw/main/docs/resources/MacaqueFaces_Metadata.png">
  <img alt="Metadata of the MacaqueFaces dataset" src="https://github.com/WildlifeDatasets/wildlife-datasets/raw/main/docs/resources/MacaqueFaces_Metadata.png">
</picture>

This particular dataset already contains cropped images of faces. Other datasets may contain uncropped images with bounding boxes or even segmentation masks.

```
d.plot_grid()
```

![](https://github.com/WildlifeDatasets/wildlife-datasets/raw/main/docs/resources/MacaqueFaces_Grid.png)

## Additional functionality

For additional functionality including mass loading, datasets splitting or evaluation metrics we refer to the [documentation](https://wildlifedatasets.github.io/wildlife-datasets/) or the [notebooks](https://github.com/WildlifeDatasets/wildlife-datasets/tree/main/notebooks).

## Citation

If you like our package, please cite our [paper](https://openaccess.thecvf.com/content/WACV2024/html/Cermak_WildlifeDatasets_An_Open-Source_Toolkit_for_Animal_Re-Identification_WACV_2024_paper.html). You may be also interested in our [SeaTurtleID](https://www.kaggle.com/datasets/wildlifedatasets/seaturtleidheads) dataset published in another [paper](https://openaccess.thecvf.com/content/WACV2024/html/Adam_SeaTurtleID2022_A_Long-Span_Dataset_for_Reliable_Sea_Turtle_Re-Identification_WACV_2024_paper.html).

```
@InProceedings{Cermak_2024_WACV,
    author    = {\v{C}erm\'ak, Vojt\v{e}ch and Picek, Luk\'a\v{s} and Adam, Luk\'a\v{s} and Papafitsoros, Kostas},
    title     = {{WildlifeDatasets: An Open-Source Toolkit for Animal Re-Identification}},
    booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
    month     = {January},
    year      = {2024},
    pages     = {5953-5963}
}
```