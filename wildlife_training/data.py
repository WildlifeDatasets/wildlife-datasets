import pycocotools.mask as mask_coco
import cv2
import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F


class ImageDataset():
    '''
    Basic image dataset for use in pytorch dataloaders.
    Uses metadata stored in dataframes.

    img_load:
        None: loads full image
        'bbox': loaded image cropped by bounding box
        'segmentation': loaded image is croppend and masked.
    '''
    def __init__(self, df, root, transform=None, img_load=None):
        self.df = df.reset_index(drop=True)
        self.root = root
        self.transform = transform
        self.img_load = img_load
        self.label, self.label_map = pd.factorize(df['identity'].values)

    def get_image(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        return img

    @property
    def num_classes(self):
        return len(self.label_map)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        data = self.df.iloc[idx]
        img = self.get_image(os.path.join(self.root, data['path']))

        if self.img_load == 'segmentation':
            mask = mask_coco.decode(data['segmentation']).astype('bool')
            img = Image.fromarray(img * mask[..., np.newaxis])
            img = img.crop((
                data['bbox'][0],
                data['bbox'][1],
                data['bbox'][0]+data['bbox'][2],
                data['bbox'][1]+data['bbox'][3]
            ))

        if self.img_load == 'bbox':
            bbox = data['bbox']
            img = img.crop((
                data['bbox'][0],
                data['bbox'][1],
                data['bbox'][0]+data['bbox'][2],
                data['bbox'][1]+data['bbox'][3]
            ))

        if self.transform:
            img = self.transform(img)

        return img, self.label[idx]


class CategoryImageDataset(ImageDataset):
    '''
    Auguments basic image dataset.
    Adds dimension with categories to image tensor.
    '''
    def __init__(self, df, root, category, transform, img_load=None):
        super().__init__(df, root, transform=transform, img_load=img_load)
        self.category, self.category_map = pd.factorize(df[category].values)

    @property
    def num_categories(self):
        return len(self.category_map)

    def __getitem__(self, idx):
        x, y = super().__getitem__(idx)
        x_category = F.one_hot(torch.tensor(self.category[idx]), num_classes=len(self.category_map))
        return (x, x_category), y



class DisjointLabelSplit():
    '''
    Splits labels such that they are disjoint.
    Scikit like splitter

    Example:
        splitter = DisjointLabelSplit(train_size=0.5, random_state=0)
        splitter.split([0, 1, 2, 3, 4, 5], ['x', 'x', 'x', 'y', 'y', 'y'])
        >>> [(array([3, 4, 5]), array([0, 1, 2]))]

    '''
    def __init__(self, train_size=0.5, random_state=0):
        self.train_size = train_size
        self.random_state = random_state

    def split(self, indexes, labels):
        np.random.seed(self.random_state)
        indexes, labels = np.array(indexes), np.array(labels)

        labels_unique = np.unique(labels)
        np.random.shuffle(labels_unique)

        i = round(len(labels_unique) * self.train_size)
        labels_train, labels_valid = labels_unique[:i], labels_unique[i:]
        return [(indexes[np.isin(labels, labels_train)], indexes[np.isin(labels, labels_valid)] )]


def split_standard(
    df,
    root,
    splitter,
    col_label = 'identity',
    create_dataset = None,
    transform_train = None,
    transform_valid = None,
    **kwargs
):
    '''
    Splits data by standard splitter class.

    args:
        df: dataframe with specification of datafiles
        root: root with data files
        splitter: spliter object with scikit like API.
        col_label: name of the column with identities/labels.
        create_dataset: function that creates suitable dataset for training.
        transform_train: transformations used for training
        transform_valid: transformations used for inference
    '''

    indexes = np.arange(len(df))
    labels = df[col_label]

    if create_dataset is None:
        create_dataset = ImageDataset

    splits = []
    i = 0
    for train, valid in splitter.split(indexes, labels):
        i = i + 1
        splits.append({
            'train': create_dataset(
                df = df.iloc[indexes[train]],
                root=root,
                transform=transform_train,
                **kwargs,
            ),

            'reference': create_dataset(
                df = df.iloc[indexes[train]],
                root=root,
                transform=transform_valid,
                **kwargs,
            ),

            'valid': create_dataset(
                df = df.iloc[indexes[valid]],
                root=root,
                transform=transform_valid,
                **kwargs,
            ),
            'name': str(i),
        })
    return splits

def split_expanding_years(
    df,
    root,
    col_year = 'year',
    create_dataset = None,
    transform_train = None,
    transform_valid = None,
    **kwargs,
):
    '''
    Splits data to 1 year expanding validation windows.

    args:
        df: dataframe with specification of datafiles
        root: root with data files
        splitter: spliter object with scikit like API.
        col_year: name of the column with years.
        create_dataset: function that creates suitable dataset for training.
        transform_train: transformations used for training
        transform_valid: transformations used for inference
    '''
    years = np.sort(np.unique(df[col_year]))

    if create_dataset is None:
        create_dataset = ImageDataset

    splits = []
    for i in range(len(years)-1):
        train_years = years[:i+1]
        valid_years = [years[i+1]]
        splits.append({
            'train': create_dataset(
                df = df[df[col_year].isin(train_years)],
                root=root,
                transform=transform_train,
                **kwargs,
            ),

            'reference': create_dataset(
                df = df[df[col_year].isin(train_years)],
                root=root,
                transform=transform_valid,
                **kwargs,
            ),

            'valid': create_dataset(
                df = df[df[col_year].isin(valid_years)],
                root=root,
                transform=transform_valid,
                **kwargs,
            ),
            'name': str(valid_years[0]),
            'years': {'valid': valid_years, 'train': train_years},
        })
    return splits