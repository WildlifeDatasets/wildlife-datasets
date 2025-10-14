# Training machine learning models

While WildlifeDatasets are not primarily intended to be used for training of deep networks, they can be easily paired with our sister project [WildlifeTools](https://github.com/WildlifeDatasets/wildlife-tools). The classes created by WildlifeDatasets may be directly plugged into WildlifeTools:

- Specify that we are interested in loading labels by `load_label=True` and `factorize_label=True`.
- Provide the `transform` attribute, which should
convert the loaded images from PIL images to torch arrays. If torch processor is used with a model, this is not necessary.

The code then looks like this:

```python
from wildlife_datasets.datasets import MacaqueFaces 
import torchvision.transforms as T

root = "data/MacaqueFaces"
transform = T.Compose([
    T.Resize([384, 384]),
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])

MacaqueFaces.get_data(root)
dataset = MacaqueFaces(
    root,
    transform=transform,
    load_label=True,
    factorize_label=True,
)
```

Now, `dataset[0]` return a tuple of the transformed image and its identity. The identity is factorized with the factorization stored in `dataset.labels_map`.

```python
dataset[0]
```
```
(tensor([[[-0.1657, -0.1657, -0.1657,  ...,  1.4783,  1.4612,  1.4612],
          [-0.1657, -0.1657, -0.1657,  ...,  1.4783,  1.4612,  1.4612],
          [-0.1657, -0.1657, -0.1657,  ...,  1.5125,  1.4954,  1.4954],
          ...,
          [ 0.1083,  0.1083,  0.1083,  ...,  1.2728,  1.2214,  1.2214],
          [ 0.1083,  0.1083,  0.1083,  ...,  1.2385,  1.1872,  1.1872],
          [ 0.1083,  0.1083,  0.1083,  ...,  1.2385,  1.1872,  1.1872]],
 
         [[ 0.1176,  0.1176,  0.1176,  ...,  1.7633,  1.7458,  1.7458],
          [ 0.1176,  0.1176,  0.1176,  ...,  1.7633,  1.7458,  1.7458],
          [ 0.1176,  0.1176,  0.1176,  ...,  1.7983,  1.7808,  1.7808],
          ...,
          [ 0.1001,  0.1001,  0.1001,  ...,  1.3957,  1.3431,  1.3431],
          [ 0.1001,  0.1001,  0.1001,  ...,  1.3606,  1.3081,  1.3081],
          [ 0.1001,  0.1001,  0.1001,  ...,  1.3606,  1.3081,  1.3081]],
 
         [[-0.2010, -0.2010, -0.2010,  ...,  1.0365,  1.0191,  1.0191],
          [-0.2010, -0.2010, -0.2010,  ...,  1.0365,  1.0191,  1.0191],
          [-0.2010, -0.2010, -0.2010,  ...,  1.0714,  1.0539,  1.0539],
          ...,
          [-0.3230, -0.3230, -0.3230,  ...,  0.7576,  0.7054,  0.7054],
          [-0.3230, -0.3230, -0.3230,  ...,  0.7228,  0.6705,  0.6705],
          [-0.3230, -0.3230, -0.3230,  ...,  0.7228,  0.6705,  0.6705]]]),
 np.int64(0))
```

Methods such as `plot_grid` work as intended.

```python
dataset.plot_grid();
```

![](images/grid_MacaqueFaces.png)

We load the [MegaDescriptor-L-384](https://huggingface.co/BVRA/MegaDescriptor-L-384) (or any other) model.

```python
import timm

model_name = "hf-hub:BVRA/MegaDescriptor-L-384"
backbone = timm.create_model(model_name, num_classes=0, pretrained=True)
```

TODO: two link missing

The class `MacaqueFaces` may then be used for example for [feature extraction](???):

```python
import numpy as np
from wildlife_tools.features import DeepFeatures
from wildlife_tools.similarity import CosineSimilarity
from wildlife_tools.inference import KnnClassifier

extractor = DeepFeatures(backbone, batch_size=4, device='cuda')

idx_train = list(range(10)) + list(range(190,200))
idx_test = list(range(10,20)) + list(range(200,210))
dataset_database = dataset.get_subset(idx_train)
dataset_query = dataset.get_subset(idx_test)
query, database = extractor(dataset_query), extractor(dataset_database)

similarity_function = CosineSimilarity()
similarity = similarity_function(query, database)

classifier = KnnClassifier(k=1, database_labels=dataset_database.labels_string)
predictions = classifier(similarity)
accuracy = np.mean(dataset_query.labels_string == predictions)
```

or for [model finetuning](???):

```python
import itertools
from torch.optim import SGD
from wildlife_tools.train import ArcFaceLoss, BasicTrainer

embedding_size = len(query[0][0])
objective = ArcFaceLoss(
    num_classes=dataset.num_classes,
    embedding_size=embedding_size,
    margin=0.5,
    scale=64
    )

params = itertools.chain(backbone.parameters(), objective.parameters())
optimizer = SGD(params=params, lr=0.001, momentum=0.9)

trainer = BasicTrainer(
    dataset=dataset,
    model=backbone,
    objective=objective,
    optimizer=optimizer,
    epochs=1,
    batch_size=8,
    device='cuda',
)

trainer.train()
```