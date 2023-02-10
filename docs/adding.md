# How to add new datasets

Adding new datasets is relatively easy. It is sufficient to create a subclass of `DatasetFactory` with the `create_catalogue` method. A simple example is

    import pandas as pd

    class Test(datasets.datasets.DatasetFactory):
        def create_catalogue(self) -> pd.DataFrame:
            df = pd.DataFrame({
                'id': [1, 2, 3, 4],
                'path': ['img1.jpg', 'img2.jpg', 'img3.jpg', 'img4.jpg'],
                'identity': ['Lukas', 'Vojta', 'Lukas', 'Vojta'],
            })
            return df

The dataframe `df` must contain columns `id` (unique id for each entry), `path` ()

TODO: finish

## Optional: including metadata


## Optional: including download