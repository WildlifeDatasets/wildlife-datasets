```python exec="true" session="run"
import contextlib, io

def run(str):
    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        eval(str)
    output = f.getvalue()
    return output
```

```python exec="true" session="run"
from wildlife_datasets import datasets, loader, metrics
import pandas as pd

df = pd.read_csv('docs/csv/MacaqueFaces.csv')
df = df.drop('Unnamed: 0', axis=1)
ds1 = datasets.MacaqueFaces('', df)
ds1.df.drop(['category'], axis=1, inplace=True)

df = pd.read_csv('docs/csv/IPanda50.csv')
df = df.drop('Unnamed: 0', axis=1)
ds2 = datasets.IPanda50('', df)

d = ds1
ds = [ds1, ds2]
```

# Testing machine learning methods

The main goal of the package is to provide a simple way for testing machine learning methods on multiple wildlife re-identification datasets.

```python
from wildlife_datasets import datasets, loader, metrics
```

## Data preparation

The datasets need to be [downloaded first](../tutorial_datasets#downloading-datasets). Assume that we have already downloaded the MacaqueFaces dataset. Then we [load it](../tutorial_datasets#working-with-multiple-datasets)

```python
d = loader.load_dataset(datasets.MacaqueFaces, 'data', 'dataframes')
```

The dataframe already contains a default split. The training dataset may be extracted by

```python exec="true" source="above" result="console" session="run"
df_train = d.df[d.df['split'] == 'train']
print(df_train) # markdown-exec: hide
```

and similarly the testing set

```python exec="true" source="above" result="console" session="run"
df_test = d.df[d.df['split'] == 'test']
print(df_test) # markdown-exec: hide
```

The training set contains 80% of the dataset. Any photo, where the animal was not recognized, are ignored for the split. Therefore, the union of the training and testing sets may be smaller than the whole dataset. It is also possible to create [custom splits](../tutorial_splits).

## Write your ML method

Now write your method. We create a prediction model predicting always the name `Dan`.

```python exec="true" source="above" session="run"
y_pred = ['Dan']*len(df_test)
```

## Evaluate the method

We implemented a [Scikit-like](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics) interface for evaluation metric. We can compute accuracy

```python exec="true" source="above" result="console" session="run"
y_true = df_test['identity']
metrics.accuracy(y_true, y_pred)
print(metrics.accuracy(y_true, y_pred)) # markdown-exec: hide
```

or any other [implemented metric](tutorial_evaluation.md).

## Mass evaluation

For mass evaluation of the developed method on wildlife re-identification datasets, we first load multiple datasets

```python
ds = loader.load_datasets(
    [datasets.IPanda50, datasets.MacaqueFaces],
    'data',
    'dataframes'
)
```

and then run the same code in a loop

```python exec="true" source="above" result="console" session="run"
for d in ds:
    df_train = d.df[d.df['split'] == 'train']
    df_test = d.df[d.df['split'] == 'test']
    
    y_pred = [df_train.iloc[0]['identity']]*len(df_test)
    y_true = df_test['identity']

    print(metrics.accuracy(y_true, y_pred))
```




