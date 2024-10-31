```python exec="true" session="run" keep_print="True"
import contextlib, io

def run(str):
    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        eval(str)
    output = f.getvalue()
    return output

def print_array(x):
    if isinstance(x, dict):
        print('{')
        for key in x.keys():
            if isinstance(x[key], str):
                print(f" '{key}': '{x[key]}'")
            else:
                print(f" '{key}': {x[key]}")
        print('}')
    elif isinstance(x, list):
        print('[')
        for y in x:
            print(f" {y}")
        print(']')
```

```python exec="true" session="run"
from wildlife_datasets import analysis, datasets, loader
import pandas as pd

df = pd.read_csv('docs/csv/MacaqueFaces.csv')
df = df.drop('Unnamed: 0', axis=1)
d = datasets.MacaqueFaces('.', df)
d.df.drop(['category'], axis=1, inplace=True)
```


# How to work with datasets

The library represents wildlife re-identification datasets and manages any operations on them such as downloads, conversions to dataframes, splitting to training and testing sets, and printing dataset summary or its metadata. We import first the required modules

```python
from wildlife_datasets import analysis, datasets, loader
```

## Downloading datasets

The majority of datasets used in this library can be downloaded fully automatically.

```python
datasets.MacaqueFaces.get_data('data/MacaqueFaces')
```

 Some of the datasets require special handling as described in a [special page](./preprocessing.md). 

## Working with one dataset

When a dataset is already downloaded, it can be loaded

```python
d = datasets.MacaqueFaces('data/MacaqueFaces')
```

Since this a subclass of the `DatasetFactory` parent class, it inherits all the methods and attributes listed in its [documentation](./reference_datasets.md). Its main component is the [pandas dataframe](./dataframe.md) of all samples

```python exec="true" source="above" result="console" session="run"
d.df
print(d.df) # markdown-exec: hide
```

The dataset can be graphically visualized by the grid plot

```python
d.plot_grid()
```

![](images/grid_MacaqueFaces.png)

or its basic numerical statistics can be printed

```python exec="true" source="above" result="console" session="run"
analysis.display_statistics(d.df)

print(run('analysis.display_statistics(d.df)')) # markdown-exec: hide
```

or [metadata](./dataframe.md#metadata) displayed

```python exec="true" source="above" result="console" session="run"
d.summary
print_array(d.summary) # markdown-exec: hide
```

## Working with multiple datasets

Since the above-mentioned way of creating the datasets always recreates the dataframe, it will be slow for larger datasets. For this reason, we provide an alternative way

```python
d = loader.load_dataset(datasets.MacaqueFaces, 'data', 'dataframes')
```

This function first checks whether `dataframes/MacaqueFaces.pkl` exists. If so, it will load the dataframe stored there, otherwise, it will create this file. Therefore, the first call of this function may be slow but the following calls are fast.

All exported datasets can be listed by

```python exec="true" source="above" result="console" session="run"
datasets.names_all
print_array(['wildlife_datasets.datasets.datasets.' + dataset_name.__name__ for dataset_name in datasets.names_all]) # markdown-exec: hide
```

!!! warning

    The following code needs to have all datasets downloaded. If you have downloaded only some of them, select the appropriate subset of `datasets.names_all`.

To work with all provided datasets, we can easily put the `load_dataset` call into a loop

```python
ds = []
for dataset_name in datasets.names_all:
    d = loader.load_dataset(dataset_name, 'data', 'dataframes')
    ds.append(d)
```

or equivalently by

```python
ds = loader.load_datasets(datasets.names_all, 'data', 'dataframes')
```
