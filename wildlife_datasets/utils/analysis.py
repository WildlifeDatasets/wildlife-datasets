import os
import numpy as np
from datasets import *
from matplotlib import pyplot as plt
from PIL import Image

info_datasets = [
    (AAUZebraFishID, {}),
    (AerialCattle2017, {}),
    (ATRW, {}),
    (BelugaID, {}),
    (BirdIndividualID, {'variant': 'source'}),
    (BirdIndividualID, {'variant': 'segmented'}),
    (CTai, {}),
    (CZoo, {}),
    (Cows2021, {}),
    (Drosophila, {}),
    (FriesianCattle2015, {}),
    (FriesianCattle2017, {}),
    (GiraffeZebraID, {}),
    (Giraffes, {}),
    (HappyWhale, {}),
    (HumpbackWhaleID, {}),
    (HyenaID2022, {}),
    (IPanda50, {}),
    (LeopardID2022, {}),
    (LionData, {}),
    (MacaqueFaces, {}),
    (NDD20, {}),
    (NOAARightWhale, {}),
    (NyalaData, {}),
    (OpenCows2020, {}),
    (SealID, {'variant': 'source'}),
    (SealID, {'variant': 'segmented'}),
    (SMALST, {}),
    (StripeSpotter, {}),
    (WhaleSharkID, {}),
    (WNIGiraffes, {}),
    (ZindiTurtleRecall, {})
]

def unique_datasets_list(datasets_list):
    _, idx = np.unique([dataset[0].__name__ for dataset in datasets_list], return_index=True)
    idx = np.sort(idx)

    datasets_list_red = []
    for i in idx:
        datasets_list_red.append(datasets_list[i])

    return datasets_list_red

def add_paths(datasets_list, root_dataset, root_dataframe):
    datasets_list_mod = []
    for dataset in datasets_list:        
        if len(dataset[1]) == 0:                        
            df_path = os.path.join(root_dataframe, dataset[0].__name__ + '.pkl')
        else:
            df_path = os.path.join(root_dataframe, dataset[0].__name__ + '_' + dataset[1]['variant'] + '.pkl')
        datasets_list_mod.append(dataset + (os.path.join(root_dataset, dataset[0].__name__),) + (df_path,))
    return datasets_list_mod




# Visualisation utils
def bbox_segmentation(bbox):
    return [bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3], bbox[0], bbox[1]+bbox[3], bbox[0], bbox[1]]

def is_annotation_bbox(ann, bbox, tol=0):
    bbox_ann = bbox_segmentation(bbox)    
    if len(ann) == len(bbox_ann):
        for x, y in zip(ann, bbox_ann):
            if abs(x-y) > tol:
                return False
    else:
        return False
    return True

def plot_image(img):
    fig, ax = plt.subplots()
    ax.imshow(img)
    plt.show()

def plot_segmentation(img, segmentation):
    if not np.isnan(segmentation).all():
        fig, ax = plt.subplots()
        ax.imshow(img)
        ax.plot(segmentation[0::2], segmentation[1::2], '--', linewidth=5, color='firebrick')
        plt.show()

def plot_bbox_segmentation(df, root, n):
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
            img = Image.open(os.path.join(root, df_red['path'].iloc[i]))
            segmentation = df_red['segmentation'].iloc[i]
            plot_segmentation(img, segmentation)
    if 'mask' in df.columns:
        df_red = df[~df['mask'].isnull()]
        for i in range(n):
            img = Image.open(os.path.join(root, df_red['mask'].iloc[i]))
            plot_image(img) 

def plot_grid(df, root, n_rows=5, n_cols=8, offset=10, img_min=100, rotate=True):
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
    display(im_grid) # TODO: remove display

def display_statistics(dataset_factory, plot_images=True, display_dataframe=True, n=2):
    # TODO: Move to utils
    df_red = dataset_factory.df.loc[dataset_factory.df['identity'] != 'unknown', 'identity']
    df_red.value_counts().reset_index(drop=True).plot()
        
    if 'unknown' in list(dataset_factory.df['identity'].unique()):
        n_identity = len(dataset_factory.df.identity.unique()) - 1
    else:
        n_identity = len(dataset_factory.df.identity.unique())
    print(f"Number of identitites          {n_identity}")
    print(f"Number of all animals          {len(dataset_factory.df)}")
    print(f"Number of identified animals   {sum(dataset_factory.df['identity'] != 'unknown')}")    
    print(f"Number of unidentified animals {sum(dataset_factory.df['identity'] == 'unknown')}")
    if 'video' in dataset_factory.df.columns:
        print(f"Number of videos               {len(dataset_factory.df[['identity', 'video']].drop_duplicates())}")
    if plot_images:
        plot_bbox_segmentation(dataset_factory.df, dataset_factory.root, n)
        plot_grid(dataset_factory.df, dataset_factory.root)
    if display_dataframe:
        display(dataset_factory.df)  # TODO: remove display