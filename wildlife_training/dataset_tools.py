import pycocotools.mask as mask_coco
import cv2
import os
import numpy as np
import pandas as pd
from PIL import Image


class BasicDataset():
    '''
    Basic image dataset for use in pytorch dataloaders.
    Uses metadata stored in dataframes.
    
    img_load:
        None: does not load images
        'full': loads full image
        'bbox': loaded image cropped by bounding box
        'segmentation': loaded image is croppend and masked.
    '''
    def __init__(self, df, root, transform=None, img_load=None):
        self.df = df
        self.root = root
        self.transform = transform
        self.img_load = img_load
        self.label, self.label_map = pd.factorize(df['identity'].values)

    def get_image(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        return img

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

        return {
            "image": img,
            "image_path": data['path'],
            "image_id": data['id'],
            "label_name": data['identity'],
            "label": self.label[idx],
        }
    

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
