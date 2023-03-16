```python exec="true"
import contextlib, io

def run(str):
    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        eval(str)
    output = f.getvalue()
    return output
```

```python exec="true"
import contextlib, io # markdown-exec: hide
 # markdown-exec: hide
def run(str): # markdown-exec: hide
    f = io.StringIO() # markdown-exec: hide
    with contextlib.redirect_stdout(f): # markdown-exec: hide
        eval(str) # markdown-exec: hide
    output = f.getvalue() # markdown-exec: hide
    return output # markdown-exec: hide
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

```python exec="true" source="above"
import contextlib, io # markdown-exec: hide
 # markdown-exec: hide
def run(str): # markdown-exec: hide
    f = io.StringIO() # markdown-exec: hide
    with contextlib.redirect_stdout(f): # markdown-exec: hide
        eval(str) # markdown-exec: hide
    output = f.getvalue() # markdown-exec: hide
    return output # markdown-exec: hide
from wildlife_datasets import datasets, splits # markdown-exec: hide
import pandas as pd # markdown-exec: hide
 # markdown-exec: hide
df = pd.read_csv('docs/csv/MacaqueFaces.csv') # markdown-exec: hide
df = df.drop('Unnamed: 0', axis=1) # markdown-exec: hide
dataset = datasets.MacaqueFaces('', df) # markdown-exec: hide
df = dataset.df # markdown-exec: hide
splitter = splits.ClosedSetSplit(df)
idx_train, idx_test = splitter.split(0.8)
df_train, df_test = df.loc[idx_train], df.loc[idx_test]
```

### Open-set split

```python
splitter = splits.OpenSetSplit(df)
idx_train, idx_test = splitter.split(0.8, 0.1)
df_train, df_test = df.loc[idx_train], df.loc[idx_test]
```

### Disjoint-set split

```python
splitter = splits.DisjointSetSplit(df)
idx_train, idx_test = splitter.split(0.2)
df_train, df_test = df.loc[idx_train], df.loc[idx_test]
```


## Splits based on time

For a proper description, see its [own page](../tutorial_splits#splits-based-on-time).

### Time-proportion split

```python exec="true" source="above"
import contextlib, io # markdown-exec: hide
 # markdown-exec: hide
def run(str): # markdown-exec: hide
    f = io.StringIO() # markdown-exec: hide
    with contextlib.redirect_stdout(f): # markdown-exec: hide
        eval(str) # markdown-exec: hide
    output = f.getvalue() # markdown-exec: hide
    return output # markdown-exec: hide
from wildlife_datasets import datasets, splits # markdown-exec: hide
import pandas as pd # markdown-exec: hide
 # markdown-exec: hide
df = pd.read_csv('docs/csv/MacaqueFaces.csv') # markdown-exec: hide
df = df.drop('Unnamed: 0', axis=1) # markdown-exec: hide
dataset = datasets.MacaqueFaces('', df) # markdown-exec: hide
df = dataset.df # markdown-exec: hide
splitter = splits.ClosedSetSplit(df) # markdown-exec: hide
idx_train, idx_test = splitter.split(0.8) # markdown-exec: hide
df_train, df_test = df.loc[idx_train], df.loc[idx_test] # markdown-exec: hide
splitter = splits.TimeProportionSplit(df)
idx_train, idx_test = splitter.split()
df_train, df_test = df.loc[idx_train], df.loc[idx_test]
```

### Time-cutoff split

```python exec="true" source="above"
import contextlib, io # markdown-exec: hide
 # markdown-exec: hide
def run(str): # markdown-exec: hide
    f = io.StringIO() # markdown-exec: hide
    with contextlib.redirect_stdout(f): # markdown-exec: hide
        eval(str) # markdown-exec: hide
    output = f.getvalue() # markdown-exec: hide
    return output # markdown-exec: hide
from wildlife_datasets import datasets, splits # markdown-exec: hide
import pandas as pd # markdown-exec: hide
 # markdown-exec: hide
df = pd.read_csv('docs/csv/MacaqueFaces.csv') # markdown-exec: hide
df = df.drop('Unnamed: 0', axis=1) # markdown-exec: hide
dataset = datasets.MacaqueFaces('', df) # markdown-exec: hide
df = dataset.df # markdown-exec: hide
splitter = splits.ClosedSetSplit(df) # markdown-exec: hide
idx_train, idx_test = splitter.split(0.8) # markdown-exec: hide
df_train, df_test = df.loc[idx_train], df.loc[idx_test] # markdown-exec: hide
splitter = splits.TimeProportionSplit(df) # markdown-exec: hide
idx_train, idx_test = splitter.split() # markdown-exec: hide
df_train, df_test = df.loc[idx_train], df.loc[idx_test] # markdown-exec: hide
splitter = splits.TimeCutoffSplit(df)
splitss, years = splitter.splits_all()
for ((idx_train, idx_test), year) in zip(splitss, years):
    df_train, df_test = df.loc[idx_train], df.loc[idx_test]
```
