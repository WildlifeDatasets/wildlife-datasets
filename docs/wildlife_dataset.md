```python exec="true" session="run"
from wildlife_datasets.datasets import MacaqueFaces
import pandas as pd

df = pd.read_csv('docs/csv/MacaqueFaces.csv')
df = df.drop('Unnamed: 0', axis=1)
root0 = 'docs/images'
dataset = MacaqueFaces(root0, df)
```

# WildlifeDataset class

The class `WildlifeDataset` is the core of the WildlifeDatasets package. It has implemnted the attributes `__len__` and `__getitem__`, which means that it can be directly [used with pytorch](./training.md). It also means that individual images can be accessed by indexing the class.

```python
from wildlife_datasets.datasets import MacaqueFaces, SeaTurtleID2022

root = 'data/MacaqueFaces'
dataset = MacaqueFaces(root)
dataset[0]
```

![](images/MacaqueFaces/Contrast/Dan/Macaque_Face_1.jpg)

This automatically loads the image at the zeroth position.
```python exec="true" source="above" session="run" result="console" 
dataset.df.iloc[0]
print(dataset.df.iloc[0]) # markdown-exec: hide
```


## Loading identities

We can load the identities by providing `load_label=True`.

```python
dataset = MacaqueFaces(root, load_label=True)
dataset[0]
```
```python exec="true" session="run" result="console" 
dataset_labels = MacaqueFaces(root0, df, load_label=True)
print(dataset_labels[0])
```

Sometimes it is necessary to have the labels converted into numerical values. For this, use `factorize_label=True`.

```python
dataset = MacaqueFaces(root, load_label=True, factorize_label=True)
dataset[0]
```
```python exec="true" session="run" result="console" 
dataset_labels = MacaqueFaces(root0, df, load_label=True, factorize_label=True)
print(dataset_labels[0])
```

## Loading bounding boxes

Multiple datasets, such as [SeaTurtleID2022](https://www.kaggle.com/datasets/wildlifedatasets/seaturtleid2022), contain bounding boxes and segmentation masks. Then it may be advantageous to load the cropped images. This is done by providing the `img_load` keyword. 

```python
dataset = SeaTurtleID2022('data/SeaTurtleID2022', img_load='bbox')
dataset[0]
```

The following grid figure shows the possible outcomes of the keyword.

![](images/loading_methods.png)


## Applying transforms

The first image has 100x100 pixels.

```python
dataset = MacaqueFaces(root)
dataset[0].size
```
```python exec="true" session="run" result="console" 
dataset = MacaqueFaces(root0, df)
print(dataset[0].size)
```

When a transformation, such as resizing images or converting it to a torch tensor, is needed, use the `transform` keyword. This example resizes the original image from 100x100 to 200x200 pixels.

```python
transform = lambda x: x.resize((200, 200))
dataset = MacaqueFaces(root, transform=transform)
dataset[0].size
```
```python exec="true" session="run" result="console" 
transform = lambda x: x.resize((200, 200))
dataset = MacaqueFaces(root0, df, transform=transform)
print(dataset[0].size)
```
