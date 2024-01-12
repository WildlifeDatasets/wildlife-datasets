# How to add new datasets

Adding new datasets is relatively easy. It is sufficient to create a subclass of `DatasetFactory` with the `create_catalogue` method. A simple example is

```python exec="true" source="above" session=""run1"
import pandas as pd
from wildlife_datasets import datasets

class Test(datasets.DatasetFactory):
    def create_catalogue(self) -> pd.DataFrame:
        df = pd.DataFrame({
            'image_id': [1, 2, 3, 4],
            'identity': ['Lukas', 'Vojta', 'Lukas', 'Vojta'],
            'path': ['img1.jpg', 'img2.jpg', 'img3.jpg', 'img4.jpg'],
        })
        return df
```

The class is then created by `Test('')`. The empty argument should point to the location where the images are stored. The dataframe can then be accessed by

```python exec="true" source="above" result="console" session=""run1"
Test('').df
print(Test('').df) # markdown-exec: hide
```

The dataframe `df` must satisfy [some requirements](../dataframe).

!!! info

    Instead of returning `df` it is better to return `self.finalize_catalogue(df)`. This function will perform [multiple checks](../reference_datasets/#datasets.datasets.DatasetFactory.finalize_catalogue) to verify the created dataframe. However, in this case, this check would fail because the specified file paths do not exist.

To incorporate the new dataset into the list of all available datasets, the [init script](https://github.com/WildlifeDatasets/wildlife-datasets/blob/main/wildlife_datasets/datasets/__init__.py) must be appropriately modified.


## Optional: including metadata

The metadata can be added by saving them in a csv file (such as [mymetadata.csv](../csv/mymetadata.csv)). Their full description is in a [separate file](../dataframe#metadata). Then they can be loaded into the class definition as a class attribute. 

```python exec="true" source="above" session=""run2"
import pandas as pd
from wildlife_datasets import datasets

class Test(datasets.DatasetFactory):
    metadata = datasets.Metadata('docs/csv/mymetadata.csv')['Test']

    def create_catalogue(self) -> pd.DataFrame:
        df = pd.DataFrame({
            'image_id': [1, 2, 3, 4],
            'identity': ['Lukas', 'Vojta', 'Lukas', 'Vojta'],
            'path': ['img1.jpg', 'img2.jpg', 'img3.jpg', 'img4.jpg'],
        })
        return df
```

The metadata can be accessed by

```python exec="true" source="above" result="console" session=""run2"
Test('').metadata
print(Test('').metadata) # markdown-exec: hide
```

## Optional: including download

Creating the download scripts should be done in the same way are the one in [download folder](https://github.com/WildlifeDatasets/wildlife-datasets/tree/main/wildlife_datasets/downloads). The simplest way is to inherit a class and write methods for download and extract.

```python
from wildlife_datasets import downloads

class Test_Downloader(downloads.Downloader):
    def _download(self):
        pass

    def _extract(self):
        pass
```

Even though the dataset could be prepared by this class, it is better to link it to the dataset.

```python
import pandas as pd
from wildlife_datasets import datasets

class Test(datasets.DatasetFactory):
    downloader = Test_Downloader
    metadata = datasets.Metadata('docs/csv/mymetadata.csv')['Test']

    def create_catalogue(self) -> pd.DataFrame:
        df = pd.DataFrame({
            'image_id': [1, 2, 3, 4],
            'identity': ['Lukas', 'Vojta', 'Lukas', 'Vojta'],
            'path': ['img1.jpg', 'img2.jpg', 'img3.jpg', 'img4.jpg'],
        })
        return df
```

## Optional: integrating into package

New datasets may be integrated into the core package by pull requests on the [Github repo](https://github.com/WildlifeDatasets/wildlife-datasets). In such a case, the dataset should be freely downloadable and both download script and metadata should be provided. The fuctions should be included in the following files:

  - `DatasetFactory` definition in [datasets.py](https://github.com/WildlifeDatasets/wildlife-datasets/blob/main/wildlife_datasets/datasets/datasets.py).
  - `Downloader` definition in [download.py](https://github.com/WildlifeDatasets/wildlife-datasets/blob/main/wildlife_datasets/downloads/downloads.py).
  - Metadata as an extension to the existing [metadata.csv](https://github.com/WildlifeDatasets/wildlife-datasets/tree/main/wildlife_datasets/datasets).

