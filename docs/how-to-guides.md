This site contains the documentation for the
**Wildlife datasets** project. 

Aim of the project is to provide comprehensive overview of datasets for wildlife 
individual identification and provide easy to use library for researchers.

## Table Of Contents

1. [Tutorials](tutorials.md)
2. [How-To Guides](how-to-guides.md)
3. [Reference](reference.md)
4. [Explanation](explanation.md)


This part of the project documentation focuses on a
**problem-oriented** approach. You'll tackle common
tasks that you might have, with the help of the code
provided in this project.



## How to use datasets

Central part of our library is `DatasetFactory` clasy, more specifically its subclasses. They represents 
wildlife datasets and manages any operations with them. `DatasetFactory` handles downloads, cleaning and spliting to subdatasets that can be used for training, for example as Pytorch Dataset. Additionally, `DatasetFactory` can create summary of the dataset and provides access to its `metadata`.

Integral component of any subclassed `DatasetFactory` is **catalogue**. It summarizes content of any dataset in easily accessable pandas DataFrames, where each row corresponds to datapoint (path to image, identity, and many more) of the dataset. 


For example, following snapshot will initialize DataFactory instance that corresponds to AAU ZebraFish ID datasets. Root is folder with downloaded images.

    dataset = AAUZebraFishID(root='data')

You can access metadata as `dataset.metadata` attributed and view summary by calling `dataset.summary()`.





## How to download datasets
Most of the datasets used in this library can be fully automatically download. 
However, some of them are require linux specific libraries. 
Please refer to `Datasets` for references.


There are two main ways how one can download dataset.


### Using scripts
You can use python scripts that are located in download module.
Each script that is available for automatic download can be used as follows:

    # Download AAU ZebraFish ID datasets
    python3 aau_zebrafish_id.py

You can also manually specify location and name of the dataset folder using optional flags:

    python3 aau_zebrafish_id.py --name 'AAUZebraFishID' --output 'data'

### Using datasets module
You can download any dataset with autmatic download when creating `DatasetFactory` instances.

    dataset = AAUZebraFishID(root='data', download=True)

Or by calling asociated downloader directly

    dataset.downloader.get_data()


## Utils
Besides datasets and downloads modules, we provide utils module that have all tools necessarily to replicate the paper XXX.