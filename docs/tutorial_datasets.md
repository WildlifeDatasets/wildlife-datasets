# How to use datasets

The central part of the library is the `DatasetFactory` class, more specifically its subclasses. They represent wildlife datasets and manage any operations on them. `DatasetFactory` handles downloads, conversions to dataframes, and splitting to training and testing set. Additionally, `DatasetFactory` can create dataset summary and provides access to its metadata.

The commands listed at this page require the following imports:

    from wildlife_datasets import datasets, analysis, loader

## Downloading datasets

Most of the datasets used in this library can be downloaded fully automatically either via a script or via a `dataset` module. However, some of them are require linux-specific libraries. 


| Dataset                | Method  |             Comments           |
|------------------------|:-------:|-------------------------------:|
| [AAUZebrafish](https://www.kaggle.com/datasets/aalborguniversity/aau-zebrafish-reid)          | AUTO    | Kaggle - CLI                   |
| [AerialCattle2017](https://data.bris.ac.uk/data/dataset/3owflku95bxsx24643cybxu3qh)          | AUTO    |                                |
| [ATRW](https://lila.science/datasets/atrw)                   | AUTO    |                                |
| [BelugaID](https://lila.science/datasets/beluga-id-2022/)              | AUTO    |                                |
| [BirdIndividualID](https://github.com/AndreCFerreira/Bird_individualID)     | MANUAL  | Manual G-Drive + Kaggle CLI    |
| [C-Tai](https://github.com/cvjena/chimpanzee_faces)       | AUTO    |                                |
| [C-Zoo](https://github.com/cvjena/chimpanzee_faces)       | AUTO    |                                |
| [Cows2021](https://data.bris.ac.uk/data/dataset/4vnrca7qw1642qlwxjadp87h7)              | AUTO    |                                |
| [Drosophila](https://github.com/j-schneider/fly_eye)             | AUTO    |                                |
| [FriesianCattle2015](https://data.bris.ac.uk/data/dataset/wurzq71kfm561ljahbwjhx9n3)   | AUTO    |                                |
| [FriesianCattle2017](https://data.bris.ac.uk/data/dataset/2yizcfbkuv4352pzc32n54371r)   | AUTO    |                                |
| [Giraffes](ftp://pbil.univ-lyon1.fr/pub/datasets/miele2021)       | AUTO    | Only linux: download                               |
| [GiraffeZebraID](https://lila.science/datasets/great-zebra-giraffe-id)       | AUTO    |                                |
| [HappyWhale](https://www.kaggle.com/competitions/happy-whale-and-dolphin)             | MANUAL  | Kaggle - Need to agree terms   |
| [HumpbackWhale](https://www.kaggle.com/competitions/humpback-whale-identification)          | MANUAL  | Kaggle - Need to agree terms   |
| [HyenaID](https://lila.science/datasets/hyena-id-2022/)               | AUTO    |                                |
| [iPanda-50](https://github.com/iPandaDateset/iPanda-50)              | AUTO    |                                |
| [LeopardID](https://lila.science/datasets/leopard-id-2022/)             | AUTO    |                                |
| [LionData](https://github.com/tvanzyl/wildlife_reidentification)              | AUTO    |                                |
| [MacaqueFaces](https://github.com/clwitham/MacaqueFaces)          | AUTO    |                                |
| [NDD20](https://doi.org/10.25405/data.ncl.c.4982342)                  | AUTO    |                                |
| [NOAARightWhale](https://www.kaggle.com/c/noaa-right-whale-recognition)        | MANUAL  | Kaggle - Need to agree terms   |
| [NyalaData](https://github.com/tvanzyl/wildlife_reidentification)             | AUTO    |                                |
| [OpenCows2020](https://data.bris.ac.uk/data/dataset/10m32xl88x2b61zlkkgz3fml17)           | AUTO    |                                |
| [SealID](https://doi.org/10.23729/0f4a3296-3b10-40c8-9ad3-0cf00a5a4a53)                 | MANUAL  | Download with single use token |
| [SeaTurtleID](https://www.kaggle.com/datasets/wildlifedatasets/seaturtleid)                 | AUTO    | Kaggle - CLI                   |
| [SMALST](https://github.com/silviazuffi/smalst)                 | AUTO    | Only linux: extract            |
| [StripeSpotter](https://code.google.com/archive/p/stripespotter/downloads)          | AUTO    | Only linux: extract            |
| [WhaleSharkID](https://lila.science/datasets/whale-shark-id)         | AUTO    |                                |
| [WNIGiraffes](https://lila.science/datasets/wni-giraffes)           | AUTO    | Only linux: download          |
| [ZindiTurtleRecall](https://zindi.africa/competitions/turtle-recall-conservation-challenge)    | AUTO    |                                |


### Using script
You can use python scripts that are located in the `download` module.
Each script can be used as follows:

    python3 macaque_faces.py

You can also manually specify the location and name of the dataset folder by the optional flags:

    python3 macaque_faces.py --name 'MacaqueFaces' --output 'data'

### Using dataset module
You can download any dataset with autmatic download when creating `DatasetFactory` instances.

    dataset = datasets.MacaqueFaces('data/MacaqueFaces', download=True)

Or by calling the asociated download script directly

    datasets.MacaqueFaces.download.get_data('data/MacaqueFaces')


## Listing datasets
All exported datasets can be listed by

    datasets.dataset_names
All these classes are subclasses of the parent class `DatasetFactory`.


## Working with one dataset
When a dataset is already downloaded, it can be loaded by
   
    dataset = datasets.MacaqueFaces('data/MacaqueFaces')

Since this a subclass of the `DatasetFactory` parent class, it inherits all the methods and attributes listed in its [documentation](reference_datasets.md). Its main component is the [pandas dataframe](../dataframe) of all samples
    
    dataset.df

and its reduced version possibly more suitable for machine learning tasks
    
    dataset.df_ml

This second dataframe removed all individual which were not known or which had only one photo.

The dataset can be graphically visualized by the grid plot

    analysis.plot_grid(dataset.df, 'data/MacaqueFaces')
or its basic numerical statistics can by printed by

    analysis.display_statistics(dataset.df)
or [metadata](../dataframe#metadata) displayed by

    dataset.metadata

## Working with multiple datasets
Since the above-mentioned way of creating the datasets always recreates the dataframe, it will be slow for larger datasets. For this reason, we provide an alternative way

    loader.load_dataset(datasets.MacaqueFaces, 'data', 'dataframes')
This function first checks whether `dataframes/MacaqueFaces.pkl` exists. If so, it will load the dataframe stored there, otherwise, it will create this file. Therefore, the first call of this function may be slow but the following calls are fast.


!!! warning

    The following code needs to have all datasets downloaded. If you have downloaded only some of them, select the appropriate subset of `datasets.dataset_names`.

To work with all provided datasets, we can easily put this function call into a loop

    ds = []
    for dataset in datasets.dataset_names:
        d = loader.load_dataset(dataset, 'data', 'dataframes')
        ds.append(d)
or equivalently by

    loader.load_datasets(datasets.dataset_names, 'data', 'dataframes')
