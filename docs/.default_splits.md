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
from wildlife_datasets import datasets, splits
import pandas as pd

df = pd.read_csv('docs/csv/MacaqueFaces.csv')
df = df.drop('Unnamed: 0', axis=1)
dataset = datasets.MacaqueFaces('', df)
df = dataset.df
```


# Default splits

The dataset already contains a default split. However, this split can be used only the the closed-set setting. This is the most common case in machine learning, where the identities (classes) in the training and testing sets coincide. Researcher may want to perform experiments on [other splits](../tutorial_splits) such as the open-set or disjoint-set splits. For comparability of methods, here we list the recommended way of creating splits. We note that the resulting splits are machine-independent and that it is possible to generate additional splits by using different seeds or proportion of the two sets.

## Splits based on identities

For a proper description, see its [own page](../tutorial_splits).

### Closed-set split

```python exec="true" source="above" session="run"
splitter = splits.ClosedSetSplit(0.8)
for idx_train, idx_test in splitter.split(df):
    df_train, df_test = df.loc[idx_train], df.loc[idx_test]
```

### Open-set split

```python exec="true" source="above" session="run"
splitter = splits.OpenSetSplit(0.8, 0.1)
for idx_train, idx_test in splitter.split(df):
    df_train, df_test = df.loc[idx_train], df.loc[idx_test]
```

### Disjoint-set split

```python exec="true" source="above" session="run"
splitter = splits.DisjointSetSplit(0.2)
for idx_train, idx_test in splitter.split(df):
    df_train, df_test = df.loc[idx_train], df.loc[idx_test]
```


## Splits based on time

For a proper description, see its [own page](../tutorial_splits#splits-based-on-time).

### Time-proportion split

```python exec="true" source="above" session="run"
splitter = splits.TimeProportionSplit()
for idx_train, idx_test in splitter.split(df):
    df_train, df_test = df.loc[idx_train], df.loc[idx_test]
```

### Time-cutoff split

```python exec="true" source="above" session="run"
splitter = splits.TimeCutoffSplitAll()
for idx_train, idx_test in splitter.split(df):
    df_train, df_test = df.loc[idx_train], df.loc[idx_test]
```
