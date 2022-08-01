import os
import pandas as pd
import hashlib
import json
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from .. import downloads
from .metadata import metadata

'''
General comments:

I would represent keypoints as they are. There is no unified notation
what they can represent (unlike segmentations and bbox) and lot of
imporant information can get lost.

I think standard attributes should always be in separate columns
 or always in 'attributes' column as dict.

We should at least provide description on how we did the data
processing and the reasoning behind it. Some datasets are especially
dificult and not obvious

'''

def find_images(root, img_extensions = ('.png', '.jpg', '.jpeg')):
    # TODO: move to dataset factory or  utils
    data = [] 
    for path, directories, files in os.walk(root):
        for file in files:
            if file.lower().endswith(tuple(img_extensions)):
                data.append({'path': os.path.relpath(path, start=root), 'file': file})
    return pd.DataFrame(data)

def create_id(string):
    # TODO: move to dataset factory or utils
    entity_id = string.apply(lambda x: hashlib.md5(x.encode()).hexdigest()[:16])
    assert len(entity_id.unique()) == len(entity_id)
    return entity_id

def bbox_segmentation(bbox):
    # TODO: Move to visualisation utils
    return [bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3], bbox[0], bbox[1]+bbox[3], bbox[0], bbox[1]]

def is_annotation_bbox(ann, bbox, tol=0):
    # TODO: Move to visualisation utils
    bbox_ann = bbox_segmentation(bbox)    
    if len(ann) == len(bbox_ann):
        for x, y in zip(ann, bbox_ann):
            if abs(x-y) > tol:
                return False
    else:
        return False
    return True

def plot_image(img):
    # TODO: Move to visualisation utils
    fig, ax = plt.subplots()
    ax.imshow(img)
    plt.show()
    
def plot_segmentation(img, segmentation):
    # TODO: Move to visualisation utils
    if not np.isnan(segmentation).all():
        fig, ax = plt.subplots()
        ax.imshow(img)
        ax.plot(segmentation[0::2], segmentation[1::2], '--', linewidth=5, color='firebrick')
        plt.show()
    
def plot_bbox_segmentation(df, root, n):
    # TODO: Move to visualisation utils
    if 'bbox' not in df.columns and 'segmentation' not in df.columns:
        for i in range(n):
            img = Image.open(os.path.join(root, df['path'][i]))
            plot_image(img)
    if 'bbox' in df.columns:
        df_red = df[~df['bbox'].isnull()]
        for i in range(n):
            img = Image.open(os.path.join(root, df_red['path'].iloc[i]))
            segmentation = bbox_segmentation(df_red['bbox'].iloc[i])
            plot_segmentation(img, segmentation)
    if 'segmentation' in df.columns:
        df_red = df[~df['segmentation'].isnull()]
        for i in range(n):
            img = Image.open(oDatasetFactory2s.path.join(root, df_red['path'].iloc[i]))
            segmentation = df_red['segmentation'].iloc[i]
            plot_segmentation(img, segmentation)
    if 'mask' in df.columns:
        df_red = df[~df['mask'].isnull()]
        for i in range(n):
            img = Image.open(os.path.join(root, df_red['mask'].iloc[i]))
            plot_image(img) 

def plot_grid(df, root, n_rows=5, n_cols=8, offset=10, img_min=100, rotate=True):
    # TODO: Move to visualisation utils
    idx = np.random.permutation(len(df))[:n_rows*n_cols]

    ratios = []
    for k in idx:
        file_path = os.path.join(root, df['path'][k])
        im = Image.open(file_path)
        ratios.append(im.size[0] / im.size[1])

    ratio = np.median(ratios)
    if ratio > 1:    
        img_w, img_h = int(img_min*ratio), int(img_min)
    else:
        img_w, img_h = int(img_min), int(img_min/ratio)

    im_grid = Image.new('RGB', (n_cols*img_w + (n_cols-1)*offset, n_rows*img_h + (n_rows-1)*offset))

    for i in range(n_rows):
        for j in range(n_cols):
            k = n_cols*i + j
            file_path = os.path.join(root, df['path'][idx[k]])

            im = Image.open(file_path)
            if rotate and ((ratio > 1 and im.size[0] < im.size[1]) or (ratio < 1 and im.size[0] > im.size[1])):
                im = im.transpose(Image.ROTATE_90)
            im.thumbnail((img_w,img_h))

            pos_x = j*img_w + j*offset
            pos_y = i*img_h + i*offset        
            im_grid.paste(im, (pos_x,pos_y))

    display(im_grid)




class DatasetFactory():
    def __init__(self, root, df, download=False, download_folder='data', **kwargs):
        self.root = root

        if download and hasattr(self, 'download'): 
            self.download.get_data(os.path.join(download_folder, self.__class__.__name__))
        if df is None:
            self.df = self.create_catalogue(**kwargs)
        else:
            self.df = df

    @classmethod
    def from_file(cls, root, df_path, save=True, overwrite=False, **kwargs):
        # TODO: reconsider pickle due to consistency - maybe json?
        if overwrite or not os.path.exists(df_path):
            df = None
            instance = cls(root, df, **kwargs)
            if save:
                instance.df.to_pickle(df_path)
        else:
            df = pd.read_pickle(df_path)
            instance = cls(root, df, **kwargs)
        return instance

    def create_catalogue(self):
        raise NotImplementedError()

    def finalize_df(self, df):
        if type(df) is dict:
            df = pd.DataFrame(df)
        df = self.reorder_df(df)
        df = self.remove_columns(df)
        self.check_unique_id(df)
        self.check_split_values(df)
        self.check_files_exist(df)
        self.check_masks_exist(df) # TODO: I would remove this or at least move is to SMALST Dataset
        return df

    def display_statistics(self, plot_images=True, display_dataframe=True, n=2):
        # TODO: Move to utils
        df_red = self.df.loc[self.df['identity'] != 'unknown', 'identity']
        df_red.value_counts().reset_index(drop=True).plot()
            
        if 'unknown' in list(self.df['identity'].unique()):
            n_identity = len(self.df.identity.unique()) - 1
        else:
            n_identity = len(self.df.identity.unique())
        print(f"Number of identitites          {n_identity}")
        print(f"Number of all animals          {len(self.df)}")
        print(f"Number of identified animals   {sum(self.df['identity'] != 'unknown')}")    
        print(f"Number of unidentified animals {sum(self.df['identity'] == 'unknown')}")
        if 'video' in self.df.columns:
            print(f"Number of videos               {len(self.df[['identity', 'video']].drop_duplicates())}")
        if plot_images:
            plot_bbox_segmentation(self.df, self.root, n)
            plot_grid(self.df, self.root)
        if display_dataframe:
            display(self.df)

    def image_sizes(self):
        # TODO: Delete this
        '''
        Return width and height of all images.

        It is slow for large datasets.
        '''
        paths = self.root + os.path.sep + self.df['path']
        data = []
        for path in paths:
            img = Image.open(path)
            data.append({'width': img.size[0], 'height': img.size[1]})
        return pd.DataFrame(data)

    def reorder_df(self, df):

        # TODO: fixed column names is never good idea. 
        # TODO: And: 'position', 'species', 'keypoints', 'date', 'video' should be in attributes.
        default_order = ['id', 'path', 'identity', 'bbox', 'segmentation', 'mask', 'position', 'species', 'keypoints', 'date', 'video', 'attributes']
        # TODO: 3 columns are always there - order them first, and rest alphabetical: df.sort_index(axis=1) + df.loc[:, cols]
        df_names = list(df.columns)
        col_names = []
        for name in default_order:
            if name in df_names:
                col_names.append(name)
        for name in df_names:
            if name not in default_order:
                col_names.append(name)
        
        df = df.sort_values('id').reset_index(drop=True)
        return df.reindex(columns=col_names)

    def remove_columns(self, df):
        # TODO: Why would i need this?
        # TODO: df.drop(columns=df.columns[df.nunique()==1], inplace=True)
        for df_name in list(df.columns):
            if df[df_name].astype('str').nunique() == 1:
                df = df.drop([df_name], axis=1)
        return df
        
    def check_unique_id(self, df):
        if len(df['id'].unique()) != len(df):
            raise(Exception('Image ID not unique.'))

    def check_split_values(self, df):
        # TODO: Fixed columns are bad idea - why do we care about this?
        allowed_values = ['train', 'test', 'val', 'database', 'query', 'unassigned']
        if 'split' in list(df.columns):
            split_values = list(df['split'].unique())
            for split_value in split_values:
                if split_value not in allowed_values:
                    raise(Exception('Split value not allowed:' + split_value))

    def check_files_exist(self, df):
        for path in df['path']:
            if not os.path.exists(os.path.join(self.root, path)):
                raise(Exception('Path does not exist:' + os.path.join(self.root, path)))

    def check_masks_exist(self, df):
        # TODO: I dont understand what exactly is at "mask" vs "segmentation"
        if 'mask' in df.columns:
            for path in df['mask']:
                if not os.path.exists(os.path.join(self.root, path)):
                    raise(Exception('Path does not exist:' + os.path.join(self.root, path)))


class Test(DatasetFactory):
    download = downloads.test
    metadata = metadata['Test']

    def create_catalogue(self):
        return pd.DataFrame([1, 2])


class AerialCattle2017(DatasetFactory):
    download = downloads.aerial_cattle_2017
    metadata = metadata['AerialCattle2017']

    def create_catalogue(self):
        data = find_images(self.root)
        folders = data['path'].str.split(os.path.sep, expand=True)

        df = {
            'id': create_id(data['file']),
            'path': data['path'] + os.path.sep + data['file'],
            'identity': folders[1].astype(int),
            'video': folders[2],
        }
        return self.finalize_df(df)


class FriesianCattle2015(DatasetFactory):
    download = downloads.friesian_cattle_2015
    metadata = metadata['FriesianCattle2015']

    def create_catalogue(self):
        data = find_images(self.root)
        folders = data['path'].str.split(os.path.sep, expand=True)
        split = folders[1].replace({'Cows-testing': 'test', 'Cows-training': 'train'})
        assert len(split.unique()) == 2

        identity = folders[2].str.strip('Cow').astype(int)

        df = {
            'id': create_id(identity.astype(str) + split + data['file']),
            'path': data['path'] + os.path.sep + data['file'],
            'identity': identity,
            'split': split
        }
        return self.finalize_df(df)
