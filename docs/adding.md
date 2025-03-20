# How to add new datasets

Adding new datasets is relatively easy. It is sufficient to create a subclass of `WildlifeDataset` with the `create_catalogue` method. A simple example is

```python exec="true" source="above" session="run1"
import pandas as pd
from wildlife_datasets import datasets

class Test(datasets.WildlifeDataset):
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

The metadata can be added by saving them in a csv file (such as [mysummary.csv](./csv/mysummary.csv)). Their full description is in a [separate file](./dataframe.md#metadata). Then they can be loaded into the class definition as a class attribute. 

```python exec="true" source="above" session="run2"
import pandas as pd
from wildlife_datasets import datasets

class Test(datasets.WildlifeDataset):
    summary = datasets.Summary('docs/csv/mysummary.csv')['Test']

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

Adding the possibility to download is achieved by adding two class methods. Examples can be taken from existing [datasets](https://github.com/WildlifeDatasets/wildlife-datasets/blob/main/wildlife_datasets/datasets/datasets.py).

```python
import pandas as pd
from wildlife_datasets import datasets

class Test(datasets.WildlifeDataset):
    summary = datasets.Summary('docs/csv/mysummary.csv')['Test']

    @classmethod
    def _download(cls):
        pass
    
    @classmethod
    def _extract(cls):
        pass

    def create_catalogue(self) -> pd.DataFrame:
        df = pd.DataFrame({
            'image_id': [1, 2, 3, 4],
            'identity': ['Lukas', 'Vojta', 'Lukas', 'Vojta'],
            'path': ['img1.jpg', 'img2.jpg', 'img3.jpg', 'img4.jpg'],
        })
        return df
```

## Optional: integrating into package

New datasets may be integrated into the core package by pull requests on the [Github repo](https://github.com/WildlifeDatasets/wildlife-datasets). In such a case, the dataset should be freely downloadable and both download script and metadata should be provided. The functions should be included in the following files:

  - `WildlifeDataset` definition in [datasets.py](https://github.com/WildlifeDatasets/wildlife-datasets/blob/main/wildlife_datasets/datasets/datasets.py).
  - Metadata as an extension to the existing [summary.csv](https://github.com/WildlifeDatasets/wildlife-datasets/tree/main/wildlife_datasets/datasets).

