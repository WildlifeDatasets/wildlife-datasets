# How to add new datasets

Adding new datasets is relatively easy. It is sufficient to create a subclass of `DatasetFactory` with the `create_catalogue` method. A simple example is

    import pandas as pd
    from wildlife_datasets import datasets

    class Test(datasets.DatasetFactory):
        def create_catalogue(self) -> pd.DataFrame:
            df = pd.DataFrame({
                'id': [1, 2, 3, 4],
                'identity': ['Lukas', 'Vojta', 'Lukas', 'Vojta'],
                'path': ['img1.jpg', 'img2.jpg', 'img3.jpg', 'img4.jpg'],
            })
            return df

The class is then created by `Test('')`. The empty argument should point to the location where the images are stored. The dataframe can then be accessed by

    Test('').df

The dataframe `df` must satisfy [some requirements](../dataframe).

!!! info

    Instead of returning `df` it is better to return `self.finalize_catalogue(df)`. This function will perform [multiple checks](../reference_datasets/#datasets.datasets.DatasetFactory.finalize_catalogue) to verify the created dataframe. However, in this case, this check would fail because the specified file paths do not exist.

To incorporate the new dataset into the list of all available datasets, the [init script](https://github.com/WildlifeDatasets/wildlife-datasets/blob/main/wildlife_datasets/datasets/__init__.py) must be appropriately modified.


## Optional: including metadata

The metadata can be added by saving them in a csv file (such as [mymetadata.csv](csv/mymetadata.csv)). Their full description is in a [separate file](../dataframe#metadata). Then they can be loaded into the class definition as a class attribute. 

    import pandas as pd
    from wildlife_datasets import datasets

    class Test(datasets.DatasetFactory):
        metadata = datasets.Metadata('mymetadata.csv')['Test']

        def create_catalogue(self) -> pd.DataFrame:
            df = pd.DataFrame({
                'id': [1, 2, 3, 4],
                'identity': ['Lukas', 'Vojta', 'Lukas', 'Vojta'],
                'path': ['img1.jpg', 'img2.jpg', 'img3.jpg', 'img4.jpg'],
            })
            return df

The metadata can be accessed by

    Test('').metadata


## Optional: including download

Creating the download scripts should be done in the same way are the one in [download folder](https://github.com/WildlifeDatasets/wildlife-datasets/tree/main/wildlife_datasets/downloads). Besides dowloading, it must also extract the files. We do not specify the exact way as downloading is different for different hosting servers. When the download is saved into the `test.py` file and placed in the above folder, it may be loaded by

    import pandas as pd
    from wildlife_datasets import datasets, downloads

    class Test(datasets.DatasetFactory):
        download = downloads.Test
        metadata = datasets.Metadata('mymetadata.csv')['Test']

        def create_catalogue(self) -> pd.DataFrame:
            df = pd.DataFrame({
                'id': [1, 2, 3, 4],
                'identity': ['Lukas', 'Vojta', 'Lukas', 'Vojta'],
                'path': ['img1.jpg', 'img2.jpg', 'img3.jpg', 'img4.jpg'],
            })
            return df