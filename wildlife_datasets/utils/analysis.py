import os
import numpy as np
import pandas as pd
import datetime
import cv2
from typing import List
from matplotlib import pyplot as plt
from PIL import Image
from ..datasets import utils

def get_image(path: str) -> Image:
    """Loads an image.

    Args:
        path (str): Path of the image.

    Returns:
        Loaded image.
    """

    # We load it with OpenCV because PIL does not apply metadata.
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img)

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
        for i in range(n):
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

    # Select indices of images to be plotted
    idx = np.random.permutation(len(df))[:n_rows*n_cols]

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
    for i in range(n_rows):
        for j in range(n_cols):
            # Load the image
            k = n_cols*i + j
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

def display_statistics(df: pd.DataFrame, unknown_name: str = '') -> None:
    """Prints statistics about the dataframe.

    Args:
        df (pd.DataFrame): A full dataframe of the data.
        unknown_name (str, optional): Name of the unknown class.
    """

    # Remove the unknown identities
    df_red = df.loc[df['identity'] != unknown_name, 'identity']
    df_red.value_counts().reset_index(drop=True).plot(xlabel='identities', ylabel='counts')
    
    # Compute the total number of identities
    if unknown_name in list(df['identity'].unique()):
        n_identity = len(df.identity.unique()) - 1
    else:
        n_identity = len(df.identity.unique())    
    n_one = len(df.groupby('identity').filter(lambda x : len(x) == 1))
    n_unidentified = sum(df['identity'] == unknown_name)

    # Print general statistics
    print(f"Number of identitites            {n_identity}")
    print(f"Number of all animals            {len(df)}")
    print(f"Number of animals with one image {n_one}")
    print(f"Number of unidentified animals   {n_unidentified}")
    print(f"Number of animals in dataframe   {len(df)-n_one-n_unidentified}")

    # Print statistics about video if present
    if 'video' in df.columns:
        print(f"Number of videos                 {len(df[['identity', 'video']].drop_duplicates())}")
    
    # Print statistics about time span if present
    if 'date' in df.columns:
        span_years = compute_span(df) / (60*60*24*365.25)
        if span_years > 1:
            print(f"Images span                      %1.1f years" % (span_years))
        elif span_years / 12 > 1:
            print(f"Images span                      %1.1f months" % (span_years * 12))
        else:
            print(f"Images span                      %1.0f days" % (span_years * 365.25))

def get_dates(dates: pd.Series, frmt: str) -> List[datetime.date]:
    """Converts the dates into the specified format.

    Args:
        dates (pd.Series): The series with dates.
        frmt (str): The desired format.

    Returns:
        The converted dates.
    """
    
    # TODO: mismatch between return types
    return np.array([datetime.datetime.strptime(date, frmt) for date in dates])

def compute_span(df: pd.DataFrame) -> float:
    """Compute the time span of the dataset.

    The span is defined as the latest time minus the earliest time of image taken.
    The times are computed separately for each individual.

    Args:
        df (pd.DataFrame): A full dataframe of the data.

    Returns:
        The span of the dataset.
    """

    # Convert the dates into timedelta
    df = df.loc[~df['date'].isnull()]
    dates = get_dates(df['date'].str[:10], '%Y-%m-%d')

    # Find the maximal span across individuals
    identities = df['identity'].unique()
    span = -np.inf
    for identity in identities:
        idx = df['identity'] == identity
        span = np.maximum(span, (max(dates[idx]) - min(dates[idx])).total_seconds())    
    return span