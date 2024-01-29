import os
import cv2
import numpy as np
import pandas as pd
from typing import Optional, List, Callable, Union
from matplotlib import pyplot as plt
from PIL import Image
from ..datasets import utils

def plot_image(img: Image) -> None:
    """Plots an image.

    Args:
        img (Image): Image to be plotted.
    """
    
    fig, ax = plt.subplots()
    ax.imshow(img)
    plt.show()
    
def plot_segmentation(img: Image, segmentation: List[float]) -> None:
    """Plots an image and its segmentation mask.

    Args:
        img (Image): Image to be plotted.
        segmentation (List[float]): Segmentation mask in the form [x1, y1, x2, y2, ...].
    """

    if not np.isnan(segmentation).all():
        fig, ax = plt.subplots()
        ax.imshow(img)
        ax.plot(segmentation[0::2], segmentation[1::2], '--', linewidth=5, color='firebrick')
        plt.show()

def plot_bbox_segmentation(df: pd.DataFrame, root: str, n: int) -> None:
    """Plots n images from the dataframe `df`.

    If bounding boxes or segmentation masks are present, it plots them as well.

    Args:
        df (pd.DataFrame): Dataframe with column `path` (relative path)
            and possibly `bbox` and `segmentation`.
        root (str): Root folder where the images are stored.
        n (int): Number of images to plot.
    """

    if 'bbox' not in df.columns and 'segmentation' not in df.columns:
        for i in range(min(n, len(df))):
            img = get_image(os.path.join(root, df['path'].iloc[i]))
            plot_image(img)
    if 'bbox' in df.columns:
        df_red = df[~df['bbox'].isnull()]
        for i in range(min(n, len(df_red))):
            img = get_image(os.path.join(root, df_red['path'].iloc[i]))
            segmentation = utils.bbox_segmentation(df_red['bbox'].iloc[i])
            plot_segmentation(img, segmentation)
    if 'segmentation' in df.columns:
        df_red = df[~df['segmentation'].isnull()]
        for i in range(min(n, len(df_red))):
            segmentation = df_red['segmentation'].iloc[i]
            if type(segmentation) == str:
                img = get_image(os.path.join(root, df_red['path'].iloc[i]))
                plot_image(img)
                img = get_image(os.path.join(root, segmentation))
                plot_image(img)
            elif type(segmentation) == list or type(segmentation) == np.ndarray:
                img = get_image(os.path.join(root, df_red['path'].iloc[i]))
                segmentation = df_red['segmentation'].iloc[i]
                plot_segmentation(img, segmentation)

def plot_grid(
        df: pd.DataFrame,
        root: str,
        *args,
        **kwargs
        ) -> Image:
    print("This function will be removed in future releases. Use d.plot_grid() instead.")
    from .. import datasets
    return datasets.DatasetFactory(root, df=df).plot_grid(*args, **kwargs)

def get_image(*args, **kwargs) -> Image:
    print("This function will be removed in future releases. Use datasets.get_image() instead.")
    from .. import datasets
    return datasets.get_image(*args, **kwargs)
