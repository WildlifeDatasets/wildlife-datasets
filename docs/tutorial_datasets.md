# How to use datasets

The central part of the library is the `DatasetFactory` class, more specifically its subclasses. They represent wildlife datasets and manage any operations on them. `DatasetFactory` handles downloads, conversions to dataframes, and splitting to training and testing set. Additionally, `DatasetFactory` can create dataset summary and provides access to its metadata.

The commands listed at this page require the following imports:

    from wildlife_datasets import datasets, analysis, loader

## Downloading datasets

Most of the datasets used in this library can be downloaded fully automatically either via a script or via a `dataset` module. However, some of them are require special handling as described in a [special page](../downloads). 

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
