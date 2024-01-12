import os
import cv2
import numpy as np
import pandas as pd
from typing import List
from matplotlib import pyplot as plt
from PIL import Image
from ..datasets import utils

def get_image(path: str, max_size: int = None) -> Image:
    """Loads an image.

    Args:
        path (str): Path of the image.
        max_size (int, optional): Maximal size of the image or None (no restriction).

    Returns:
        Loaded image.
    """

    # We load it with OpenCV because PIL does not apply metadata.
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    if max_size is not None:
        w, h = img.size
        if max(w, h) > max_size:
            c = max_size / max(w, h)
            img = img.resize((int(c*w), int(c*h)))
    return img

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
        n_rows: int = 5,
        n_cols: int = 8,
        offset: float = 10,
        img_min: float = 100,
        rotate: bool = True
        ) -> Image:
    """Plots a grid of size (n_rows, n_cols) with images from the dataframe.

    Args:
        df (pd.DataFrame): Dataframe with column `path` (relative path).
        root (str): Root folder where the images are stored. 
        n_rows (int, optional): The number of rows in the grid.
        n_cols (int, optional): The number of columns in the grid.
        offset (float, optional): The offset between images.
        img_min (float, optional): The minimal size of the plotted images.
        rotate (bool, optional): Rotates the images to have the same orientation.

    Returns:
        The plotted grid.
    """

    if len(df) == 0:
        return None
    
    # Select indices of images to be plotted
    n = min(len(df), n_rows*n_cols)
    idx = np.random.permutation(len(df))[:n]

    # Load images and compute their ratio
    ratios = []
    for k in idx:
        file_path = os.path.join(root, df.iloc[k]['path'])
        im = get_image(file_path)
        ratios.append(im.size[0] / im.size[1])

    # Get the size of the images after being resized
    ratio = np.median(ratios)
    if ratio > 1:    
        img_w, img_h = int(img_min*ratio), int(img_min)
    else:
        img_w, img_h = int(img_min), int(img_min/ratio)

    # Create an empty image grid
    im_grid = Image.new('RGB', (n_cols*img_w + (n_cols-1)*offset, n_rows*img_h + (n_rows-1)*offset))

    # Fill the grid image by image
    for k in range(n):
        i = k // n_cols
        j = k % n_cols

        # Load the image
        file_path = os.path.join(root, df.iloc[idx[k]]['path'])
        im = get_image(file_path)

        # Possibly rotate the image
        if rotate and ((ratio > 1 and im.size[0] < im.size[1]) or (ratio < 1 and im.size[0] > im.size[1])):
            im = im.transpose(Image.ROTATE_90)

        # Rescale the image
        im.thumbnail((img_w,img_h))

        # Place the image on the grid
        pos_x = j*img_w + j*offset
        pos_y = i*img_h + i*offset        
        im_grid.paste(im, (pos_x,pos_y))
    return im_grid
