# How to add new datasets

Adding new datasets to WildlifeDatasets is easy. It is sufficient to create a subclass of `WildlifeDataset` with the `create_catalogue` method. A simple example is

```python exec="true" source="above" session="run1"
import pandas as pd
from wildlife_datasets.datasets import WildlifeDataset

class Test(WildlifeDataset):
    def create_catalogue(self) -> pd.DataFrame:
        df = pd.DataFrame({
            'image_id': [1, 2, 3, 4],
            'identity': ['Lukas', 'Vojta', 'Lukas', 'Vojta'],
            'path': ['img1.jpg', 'img2.jpg', 'img3.jpg', 'img4.jpg'],
        })
        return df
```

The class is then created by `Test('.')`. The empty argument should point to the location where the images are stored. The dataframe can then be accessed by

```python exec="true" source="above" result="console" session="run1"
Test('.').df
print(Test('.').df) # markdown-exec: hide
```

The dataframe `df` must satisfy [some requirements](./dataframe.md).

!!! info

    Instead of returning `df` it is better to return `self.finalize_catalogue(df)`. This function will perform [multiple checks](./reference_datasets.md/#datasets.datasets.WildlifeDataset.finalize_catalogue) to verify the created dataframe. However, in this case, this check would fail because the specified file paths do not exist.

To incorporate the new dataset into the list of all available datasets, the [init script](https://github.com/WildlifeDatasets/wildlife-datasets/blob/main/wildlife_datasets/datasets/__init__.py) must be appropriately modified.


## Optional: including metadata

The metadata can be added by adding a dictionary as a class attribute. Its description is in a [separate file](./dataframe.md#metadata).

```python exec="true" source="above" session="run2"
import pandas as pd
from wildlife_datasets.datasets import WildlifeDataset

summary = {
    'reported_n_total': 4,
    'reported_n_individuals': 2,
}

class Test(WildlifeDataset):
    summary = summary

    def create_catalogue(self) -> pd.DataFrame:
        df = pd.DataFrame({
            'image_id': [1, 2, 3, 4],
            'identity': ['Lukas', 'Vojta', 'Lukas', 'Vojta'],
            'path': ['img1.jpg', 'img2.jpg', 'img3.jpg', 'img4.jpg'],
        })
        return df
```

The metadata can be accessed by

```python exec="true" source="above" result="console" session="run2"
Test('.').summary
print(Test('.').summary) # markdown-exec: hide
```

## Optional: including download

Adding the possibility to download is achieved by adding class methods `_download` and `_extract`. The simplest way is to use the predefined classes `DownloadKaggle`, `DownloadURL` and `DownloadHuggingFace`. In the multiple inheritence, `WildlifeDatasets` must always be inherited as last. Examples can be taken from existing [datasets](https://github.com/WildlifeDatasets/wildlife-datasets/blob/main/wildlife_datasets/datasets/).

### Downloads from Kaggle (recommended)

[Kaggle](https://www.kaggle.com/) supports free storage for competitions and datasets with fast download. The web address have two possible styles:

```
https://www.kaggle.com/competitions/animal-clef-2025
https://www.kaggle.com/datasets/wildlifedatasets/seaturtleid2022
```

It is sufficient to provided `kaggle_url` and `kaggle_type` as in the two examples below. The loaded class `DownloadKaggle` will add all the required functions for downloads and it is sufficient to write the `create_catalogue` function mentioned above.

```python
from wildlife_datasets.datasets import DownloadKaggle, WildlifeDataset

class AnimalCLEF2025(DownloadKaggle, WildlifeDataset):    
    kaggle_url = 'animal-clef-2025'
    kaggle_type = 'competitions'

class SeaTurtleID2022(DownloadKaggle, WildlifeDataset):
    kaggle_url = 'wildlifedatasets/seaturtleid2022'
    kaggle_type = 'datasets'
```

### Downloads from URL

Datasets may be stored at a private server. When there is a single file to download and extract, it is stored in the `url` and `archive` attributes. The latter is usually the last part of the former. Whenever multiple files need to be downloaded, they are saved in the `downloads` attribute as the examples below show.

```python
from wildlife_datasets.datasets import DownloadURL, WildlifeDataset

class CTai(DownloadURL, WildlifeDataset):
    url = 'https://github.com/cvjena/chimpanzee_faces/archive/refs/heads/master.zip'
    archive = 'master.zip'

class MacaqueFaces(DownloadURL, WildlifeDataset):
    downloads = [
        ('https://github.com/clwitham/MacaqueFaces/raw/master/ModelSet/MacaqueFaces.zip', 'MacaqueFaces.zip'),
        ('https://github.com/clwitham/MacaqueFaces/raw/master/ModelSet/MacaqueFaces_ImageInfo.csv', 'MacaqueFaces_ImageInfo.csv'),
    ]
```

### Downloads from HuggingFace (not recommended)

When the dataset is saved to [HuggingFace](https://huggingface.co/), it is sufficient to provide the `hf_url` attribute. However, due to different way of storing the images for these datasets, it may be necessary to overwrite the method `get_image` as showed, for example, in [this file](https://github.com/WildlifeDatasets/wildlife-datasets/blob/main/wildlife_datasets/datasets/chicks4free_id.py).

```python
from wildlife_datasets.datasets import DownloadHuggingFace, WildlifeDataset

class Chicks4FreeID(DownloadHuggingFace, WildlifeDataset):
    hf_url = 'dariakern/Chicks4FreeID'
```


## Optional: integrating into package

New datasets may be integrated into the core package by pull requests on the [Github repo](https://github.com/WildlifeDatasets/wildlife-datasets). In such a case, the dataset should be freely downloadable and both download script and metadata should be provided. The added dataset should be placed into [this folder](https://github.com/WildlifeDatasets/wildlife-datasets/blob/main/wildlife_datasets/datasets).
  
