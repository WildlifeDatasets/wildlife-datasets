# Segmentation Masks

WildlifeDatasets provides segmentation masks that separate animals from their surroundings. 
These masks can be used to generate foreground and background views, apply image augmentations, or evaluate whether identification models rely on the animal itself or contextual information.

The library supports both original dataset annotations and collection of generated segmentation masks, developed as part of [PAPER REFERENCE — TO BE ADDED]. 
Both sources are accessible through the same interface, making it easy to work with masks across datasets.

## Loading Segmentation Masks

Segmentation masks can be access once downloaded using the standard dataset preparation procedure and by setting `masks=True`:

```python
from wildlife_datasets import datasets

root = "data/NyalaData"
datasets.NyalaData.get_data(root, masks=True)

dataset = datasets.NyalaData(root)
```

If the dataset images are already downloaded, only the missing masks will we _prepared_. 
Once available, masks are automatically loaded with the dataset.

## Accessing Segmentation Masks

Once the dataset is loaded, images and their corresponding segmentation masks can be accessed directly:

```python
item = dataset[0]

image = item.image
mask = item.mask
```

The image is returned as an RGB PIL image, while the mask is a Boolean NumPy array with the same height and width. 
Pixels belonging to the animal are marked `True`, and background pixels are marked `False`.
If a segmentation mask is unavailable, `item.mask` returns `None`. 

> **Note**: Depending on the dataset, masks may cover the entire animal or only a specific body part.

## Accessing Images with Masks

To only select images with available masks, use `has_mask()`:

```python
positions = [i for i in range(len(dataset)) if dataset.has_mask(i)]
item = dataset[positions[0]]
```

> **Note**: Accessing masks requires the default img_load="full" and transform=None settings. When using load_label=True, retrieve samples as item, identity = dataset[i].

## Generating Image Views

Segmentation masks can be used to isolate the animal, remove it from the scene, or extract its silhouette. WildlifeDatasets provides these views directly through the dataset item:

```python
foreground = item.foreground
background = item.background
silhouette = item.silhouette
```

| Property | Description |
|---|---|
| `.foreground` | Animal foreground with the background removed. |
| `.background` | Background with the animal removed. |
| `.shortcut` | Same as `.background`. |
| `.silhouette` | Black silhouette of the animal. |
| `.cfr_views` | Dictionary containing the full, foreground, shortcut, and silhouette views. |

All views are returned as RGB PIL images. Removed regions are replaced with a neutral color corresponding to the ImageNet RGB mean `(123, 116, 103)`.

For [CFR and SSI evaluation](cfr.md), all four views can be accessed together:

```python
views = item.cfr_views

full = views["full"]
foreground = views["foreground"]
shortcut = views["shortcut"]
silhouette = views["silhouette"]
```

> **Note:** that removing the animal leaves its outline visible in the background view. The silhouette view allows this shape information to be evaluated separately.

## Using Masks for Image Augmentation

Segmentation masks can also be used to create custom image augmentations. For example, the following code replaces the background with white while preserving the animal:

```python
import numpy as np
from PIL import Image

image = item.image
mask = item.mask

pixels = np.asarray(image).copy()
pixels[~mask] = [255, 255, 255]

augmented_image = Image.fromarray(pixels)
```

When applying spatial transformations, such as flipping, cropping, or rotating, the same transformation should be applied to both the image and its mask.
For example, to flip both horizontally:

```python
from PIL import ImageOps

flipped_image = ImageOps.mirror(image)
flipped_mask = np.fliplr(mask).copy()
```

When resizing masks, use nearest-neighbor interpolation to preserve their binary values.

## Background Inpainting

Removing an animal with a constant color leaves an artificial region in the image. 
In case you want/need a more natural background, we provides an optional Stable Diffusion inpainting interface that reconstructs the scene behind the animal.

First, install the additional dependencies:

```bash
pip install "wildlife-datasets[inpainting]"
```

Then initialize the background filler and apply it to a dataset item:

```python
from wildlife_datasets import preparation

fill = preparation.StableDiffusionBackgroundFill()

background = fill(item)
background.save("background.png")
```

The filler uses the image and its segmentation mask to reconstruct the region occupied by the animal. 
It returns an RGB PIL image at the original resolution, preserving pixels outside the expanded mask.
You can (and you should) provide a prompt describing the scene:

```python
background = fill(
    item,
    prompt="Fill the masked part, fit it so it matches the background.",
    seed=42,
)
```

The model is downloaded automatically on first use and cached locally. 
It is loaded once and reused for subsequent images.
By default, CUDA and Apple MPS acceleration are supported, with CPU as a fallback.

Additional options include `device`, `model`, `resolution`, `num_inference_steps`, `guidance_scale`, and `mask_dilation`. 
A preloaded Diffusers pipeline can also be supplied through `pipeline=pipe`.

> **Note:** Inpainting can introduce artificial objects or textures, so generated backgrounds should be inspected before use. 
Unlike the constant-fill views used in standard [CFR and SSI evaluation](cfr.md), inpainted backgrounds contain generated visual information and should be treated as a separate experimental condition.