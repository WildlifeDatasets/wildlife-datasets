import os
import pandas as pd
import hashlib
import json
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image


def find_images(root, img_extensions = ('.png', '.jpg', '.jpeg')):
    data = []
    for path, directories, files in os.walk(root):
        for file in files:
            if file.lower().endswith(tuple(img_extensions)):
                data.append({'path': os.path.relpath(path, start=root), 'file': file})
    return pd.DataFrame(data)

def create_id(string):
    entity_id = string.apply(lambda x: hashlib.md5(x.encode()).hexdigest()[:16])
    assert len(entity_id.unique()) == len(entity_id)
    return entity_id

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

    display(im_grid)



class DatasetFactory():
    def __init__(self, root, df, save=True, **kwargs):
        self.root = root
        if df is None:
            self.df = self.create_catalogue(**kwargs)
        else:
            self.df = df

    @classmethod
    def from_file(cls, root, df_path, save=True, overwrite=False, **kwargs):
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
        self.check_masks_exist(df)
        return df

    def display_statistics(self, plot_images=True, display_dataframe=True, n=2):
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
        default_order = ['id', 'path', 'identity', 'bbox', 'segmentation', 'mask', 'position', 'species', 'keypoints', 'date', 'video', 'attributes']
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
        for df_name in list(df.columns):
            if df[df_name].astype('str').nunique() == 1:
                df = df.drop([df_name], axis=1)
        return df
        
    def check_unique_id(self, df):
        if len(df['id'].unique()) != len(df):
            raise(Exception('Image ID not unique.'))

    def check_split_values(self, df):
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
        if 'mask' in df.columns:
            for path in df['mask']:
                if not os.path.exists(os.path.join(self.root, path)):
                    raise(Exception('Path does not exist:' + os.path.join(self.root, path)))





class DatasetFactoryWildMe(DatasetFactory):
    def create_catalogue_wildme(self, prefix, year):
        path_json = os.path.join(prefix + '.coco', 'annotations', 'instances_train' + str(year) + '.json')
        path_images = os.path.join(prefix + '.coco', 'images', 'train' + str(year))

        # Load annotations JSON file
        with open(os.path.join(self.root, path_json)) as file:
            data = json.load(file)

        # Check whether segmentation is different from a box
        for ann in data['annotations']:
            if len(ann['segmentation']) != 1:
                raise(Exception('Wrong number of segmentations'))
            
        create_dict = lambda i: {'identity': i['name'], 'bbox': i['bbox'], 'image_id': i['image_id'], 'category_id': i['category_id'], 'segmentation': i['segmentation'][0]}
        df_annotation = pd.DataFrame([create_dict(i) for i in data['annotations']])

        create_dict = lambda i: {'file_name': i['file_name'], 'image_id': i['id'], 'date': i['date_captured']}
        df_images = pd.DataFrame([create_dict(i) for i in data['images']])

        species = pd.DataFrame(data['categories'])
        species = species.rename(columns={'id': 'category_id', 'name': 'species'})

        df = pd.merge(df_annotation, species, how='left', on='category_id')
        df = pd.merge(df, df_images, how='left', on='image_id')
        df['path'] = path_images + os.path.sep + df['file_name']
        df['id'] = range(len(df))    
        df.loc[df['identity'] == '____', 'identity'] = 'unknown'

        # Remove segmentations which are the same as bounding boxes
        ii = []
        for i in range(len(df)):
            ii.append(is_annotation_bbox(df.loc[i].segmentation, df.loc[i].bbox, tol=2))
        df.loc[ii, 'segmentation'] = np.nan

        df = df.drop(['image_id', 'file_name', 'supercategory', 'category_id'], axis=1)
        if len(df['species'].unique()) == 1:
            df = df.drop(['species'], axis=1)
        return self.finalize_df(df)



class AAUZebraFishID(DatasetFactory):
    licenses = 'Attribution 4.0 International (CC BY 4.0)'
    licenses_url = 'https://creativecommons.org/licenses/by/4.0/'
    url = 'https://www.kaggle.com/datasets/aalborguniversity/aau-zebrafish-reid'
    cite = 'bruslund2020re'
    animals = ('zebrafish')
    real_animals = True    
    year = 2020
    reported_n_total = 6672
    reported_n_identified = 6672
    reported_n_photos = 2224
    reported_n_individuals = 6 
    wild = False
    clear_photos = True
    pose = 'double' # from either side
    unique_pattern = False
    from_video = True
    full_frame = True
    
    def create_catalogue(self):
        data = pd.read_csv(os.path.join(self.root, 'annotations.csv'), sep=';')

        columns_bbox = [
            'Upper left corner X',
            'Upper left corner Y',
            'Lower right corner X',
            'Lower right corner Y',
        ]
        bbox = data[columns_bbox].to_numpy()
        bbox[:,2] = bbox[:,2] - bbox[:,0]
        bbox[:,3] = bbox[:,3] - bbox[:,1]
        bbox = pd.Series(list(bbox))

        attributes = data['Right,Turning,Occlusion,Glitch'].str.split(',', expand=True)
        attributes.columns = ['Right', 'Turning', 'Occlusion', 'Glitch']
        attributes = attributes.astype(bool).to_dict(orient='index')

        video = data['Filename'].str.split('_',  expand=True)[0]
        video = video.astype('category').cat.codes

        df = {
            'id': create_id(data['Object ID'].astype(str) + data['Filename']),
            'path': 'data' + os.path.sep + data['Filename'],
            'identity': data['Object ID'],
            'video': video,
            'bbox': bbox,
            'attributes': attributes,
        }
        return self.finalize_df(df)



class AerialCattle2017(DatasetFactory):
    licenses = 'Non-Commercial Government Licence for public sector information'
    licenses_url = 'https://www.nationalarchives.gov.uk/doc/non-commercial-government-licence/version/2/'
    url = 'https://data.bris.ac.uk/data/dataset/3owflku95bxsx24643cybxu3qh'
    cite = 'andrew2017visual'
    animals = ('Friesian cattle')
    real_animals = True    
    year = 2017
    reported_n_total = 46340
    reported_n_identified = 46340
    reported_n_photos = 46340
    reported_n_individuals = 23
    wild = False
    clear_photos = True
    pose = 'single' # from the top
    unique_pattern = True
    from_video = True
    full_frame = False

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



class ATRW(DatasetFactory):
    licenses = 'Attribution-NonCommercial-ShareAlike 4.0 International'
    licenses_url = 'https://creativecommons.org/licenses/by-nc-sa/4.0/'
    url = 'https://lila.science/datasets/atrw'
    cite = 'li2019atrw'
    animals = ('amur tiger')
    real_animals = True    
    year = 2019
    reported_n_total = 9496
    reported_n_identified = 3649
    reported_n_photos = 8076
    reported_n_individuals = 92
    wild = False # Chinese zoos
    clear_photos = False # occlussions, shadows
    pose = 'double' # from either side
    unique_pattern = True
    from_video = True
    full_frame = True

    def create_catalogue(self):
        ids = pd.read_csv(os.path.join(self.root, 'atrw_anno_reid_train', 'reid_list_train.csv'),
                        names=['identity', 'path'],
                        header=None
                        )
        ids['id'] = ids['path'].str.split('.', expand=True)[0].astype(int)
        ids['split'] = 'train'

        with open(os.path.join(self.root, 'atrw_anno_reid_train', 'reid_keypoints_train.json')) as file:
            keypoints = json.load(file)

        df_keypoints = {
            'path': pd.Series(keypoints.keys()),
            'keypoints': pd.Series(list(pd.DataFrame([keypoints[key] for key in keypoints.keys()]).to_numpy())),
        }
        data = pd.DataFrame(df_keypoints)

        df_train = pd.merge(ids, data, on='path')
        df_train['path'] = 'atrw_reid_train' + os.path.sep + 'train' + os.path.sep + df_train['path']

        # Add testing data 1
        with open(os.path.join(self.root, 'eval_script', 'ATRWEvalScript-main', 'annotations', 'gt_test_plain.json')) as file:
            identity = json.load(file)
        identity = pd.DataFrame(identity)

        ids = pd.read_csv(os.path.join(self.root, 'atrw_anno_reid_test', 'reid_list_test.csv'),
                        names=['path'],
                        header=None
                        )
        ids['id'] = ids['path'].str.split('.', expand=True)[0].astype(int)
        ids['split'] = 'test'

        ids = pd.merge(ids, identity, left_on='id', right_on='imgid')
        ids = ids.drop(['query', 'frame', 'imgid'], axis=1)
        ids.rename(columns = {'entityid': 'identity'}, inplace = True)

        with open(os.path.join(self.root, 'atrw_anno_reid_test', 'reid_keypoints_test.json')) as file:
            keypoints = json.load(file)

        df_keypoints = {
            'path': pd.Series(keypoints.keys()),
            'keypoints': pd.Series(list(pd.DataFrame([keypoints[key] for key in keypoints.keys()]).to_numpy())),
        }
        data = pd.DataFrame(df_keypoints)

        df_test1 = pd.merge(ids, data, on='path')
        df_test1['path'] = 'atrw_reid_test' + os.path.sep + 'test' + os.path.sep + df_test1['path']

        # Add testing data 2
        with open(os.path.join(self.root, 'eval_script', 'ATRWEvalScript-main', 'annotations', 'gt_test_wild.json')) as file:
            identity = json.load(file)

        ids = find_images(os.path.join(self.root, 'atrw_detection_test', 'test'))
        ids['imgid'] = ids['file'].str.split('.', expand=True)[0].astype('int')

        entries = []
        for key in identity.keys():
            for entry in identity[key]:
                bbox = [entry['bbox'][0], entry['bbox'][1], entry['bbox'][2]-entry['bbox'][0], entry['bbox'][3]-entry['bbox'][1]]
                entries.append({'imgid': int(key), 'bbox': bbox, 'identity': entry['eid']})
        entries = pd.DataFrame(entries)

        df_test2 = pd.merge(ids, entries, on='imgid')
        df_test2['path'] = 'atrw_detection_test' + os.path.sep + 'test' + os.path.sep + df_test2['file']
        df_test2['id'] = df_test2['imgid'].astype(str) + '_' + df_test2['identity'].astype(str)
        df_test2['split'] = 'test'
        df_test2 = df_test2.drop(['file', 'imgid'], axis=1)

        df = pd.concat([df_train, df_test1, df_test2])
        df['id'] = create_id(df.id.astype(str))
        return self.finalize_df(df)

    
    
class BelugaID(DatasetFactoryWildMe):
    licenses = 'Attribution-NonCommercial-NoDerivs License'
    licenses_url = 'http://creativecommons.org/licenses/by-nc-nd/2.0/'
    url = 'https://lila.science/datasets/beluga-id-2022/'
    cite = 'belugaid'
    animals = ('beluga whale')
    real_animals = True    
    year = 2022
    reported_n_total = 5902
    reported_n_identified = 5902
    reported_n_photos = 5902
    reported_n_individuals = 788
    wild = True
    clear_photos = True
    pose = 'single' # from the top
    unique_pattern = False
    from_video = False
    full_frame = False

    def create_catalogue(self):
        return self.create_catalogue_wildme('beluga', 2022)



class BirdIndividualID(DatasetFactory):
    licenses = None
    licenses_url = None
    url = 'https://github.com/AndreCFerreira/Bird_individualID'
    cite = 'ferreira2020deep'
    animals = ('sociable weaver', 'great tit', 'zebra finch')
    real_animals = True    
    year = 2019
    reported_n_total = 27038+7605+16000 # plus some test
    reported_n_identified = 27038+7605+16000 # plus some test
    reported_n_photos = 27038+7605+16000 # plus some test
    reported_n_individuals = 30+10+10 # plus some test
    wild = False
    clear_photos = True
    pose = 'single' # from the top
    unique_pattern = False
    from_video = False
    full_frame = True

    def create_catalogue(self, variant='source'):
        if variant == 'source':
            prefix1 = 'Original_pictures'
            prefix2 = 'IndividualID'
        elif variant == 'segmented':
            prefix1 = 'Cropped_pictures'
            prefix2 = 'IndividuaID'
        else:
            raise ValueError(f'Variant {variant} is not valid')

        path = os.path.join(self.root, prefix1, prefix2)
        data = find_images(path)
        folders = data['path'].str.split(os.path.sep, expand=True)

        # Remove images with multiple labels
        idx = folders[2].str.contains('_')
        data = data.loc[~idx]
        folders = folders.loc[~idx]

        # Remove some problems with the sociable_weavers/Test_dataset
        if folders.shape[1] == 4:
            idx = folders[3].isnull()
            folders.loc[~idx, 2] = folders.loc[~idx, 3]

        split = folders[1].replace({'Test_datasets': 'test', 'Test': 'test', 'Train': 'train', 'Val':'val'})
        identity = folders[2]
        species = folders[0]

        df1 = {    
            'id': create_id(split + data['file']),
            'path': prefix1 + os.path.sep + prefix2 + os.path.sep + data['path'] + os.path.sep + data['file'],
            'identity': identity,
            'species': species,
            'split': split,
        }

        # Add images without labels
        path = os.path.join(self.root, prefix1, 'New_birds')
        data = find_images(path)
        species = data['path']

        df2 = {    
            'id': create_id(data['file']),
            'path': prefix1 + os.path.sep + 'New_birds' + os.path.sep + data['path'] + os.path.sep + data['file'],
            'identity': 'unknown',
            'species': species,
            'split': 'unassigned',
        }

        return self.finalize_df(pd.concat([pd.DataFrame(df1), pd.DataFrame(df2)]))


class CTai(DatasetFactory):
    licenses = None
    licenses_url = None
    url = 'https://github.com/cvjena/chimpanzee_faces'
    cite = 'freytag2016chimpanzee'
    animals = ('chimpanzee')
    real_animals = True    
    year = 2016
    reported_n_total = 5078
    reported_n_identified = 5078
    reported_n_photos = 5078
    reported_n_individuals = 78 
    wild = True
    clear_photos = True
    pose = 'single' # from the front
    unique_pattern = False
    from_video = False
    full_frame = False

    def create_catalogue(self):
        replace_names = [
            ('Adult', 'unknown'),
            ('Akouba', 'Akrouba'),
            ('Freddy', 'Fredy'),
            ('Ibrahiim', 'Ibrahim'),
            ('Liliou', 'Lilou'),
            ('Wapii', 'Wapi'),
            ('Woodstiock', 'Woodstock')
        ]
            
        path = os.path.join('chimpanzee_faces-master', 'datasets_cropped_chimpanzee_faces', 'data_CTai',)
        data = pd.read_csv(os.path.join(self.root, path, 'annotations_ctai.txt'), header=None, sep=' ')
        
        data = data.rename(columns={5: 'age', 7: 'age_group', 9: 'gender'})
        attributes = data[['age', 'age_group', 'gender']].to_dict(orient='index')
        keypoints = data[[11, 12, 14, 15, 17, 18, 20, 21, 23, 24]].to_numpy()
        keypoints[np.isinf(keypoints)] = np.nan
        keypoints = pd.Series(list(keypoints))
        
        df = {
            'id': pd.Series(range(len(data))),
            'path': path + os.path.sep + data[1],
            'identity': data[3],
            'keypoints': keypoints,
            'attributes': attributes
        }
        df = pd.DataFrame(df)
        for replace_tuple in replace_names:
            print(replace_tuple)
            df['identity'] = df['identity'].replace({replace_tuple[0]: replace_tuple[1]})
        return self.finalize_df(df)



class CZoo(DatasetFactory):
    licenses = None
    licenses_url = None
    url = 'https://github.com/cvjena/chimpanzee_faces'
    cite = 'freytag2016chimpanzee'
    animals = ('chimpanzee')
    real_animals = True    
    year = 2016
    reported_n_total = 2109
    reported_n_identified = 2109
    reported_n_photos = 2109
    reported_n_individuals = 24
    wild = False
    clear_photos = True
    pose = 'single' # from the front
    unique_pattern = False
    from_video = False
    full_frame = False

    def create_catalogue(self):
        path = os.path.join('chimpanzee_faces-master', 'datasets_cropped_chimpanzee_faces', 'data_CZoo',)
        data = pd.read_csv(os.path.join(self.root, path, 'annotations_czoo.txt'), header=None, sep=' ')

        data = data.rename(columns={5: 'age', 7: 'age_group', 9: 'gender'})
        attributes = data[['age', 'age_group', 'gender']].to_dict(orient='index')
        keypoints = data[[11, 12, 14, 15, 17, 18, 20, 21, 23, 24]].to_numpy()
        keypoints[np.isinf(keypoints)] = np.nan
        keypoints = pd.Series(list(keypoints))
        
        df = {
            'id': pd.Series(range(len(data))),
            'path': path + os.path.sep + data[1],
            'identity': data[3],
            'keypoints': keypoints,
            'attributes': attributes
        }
        return self.finalize_df(df)



class Cows2021(DatasetFactory):
    licenses = 'Non-Commercial Government Licence for public sector information'
    licenses_url = 'https://www.nationalarchives.gov.uk/doc/non-commercial-government-licence/version/2/'
    url = 'https://data.bris.ac.uk/data/dataset/4vnrca7qw1642qlwxjadp87h7'
    cite = 'gao2021towards'
    animals = ('cow')
    real_animals = True    
    year = 2021
    reported_n_total = 13784
    reported_n_identified = 13784
    reported_n_photos = 13784
    reported_n_individuals = 181
    wild = False
    clear_photos = True
    pose = 'single' # from the top
    unique_pattern = True
    from_video = True
    full_frame = False

    def create_catalogue(self):
        data = find_images(self.root)
        folders = data['path'].str.split(os.path.sep, expand=True)

        ii = (folders[2] == 'Identification') & (folders[3] == 'Test')
        folders = folders.loc[ii]
        data = data.loc[ii]

        df = {
            'id': create_id(data['file']),
            'path': data['path'] + os.path.sep + data['file'],
            'identity': folders[4].astype(int),
        }
        return self.finalize_df(df)



class Drosophila(DatasetFactory):
    licenses = None
    licenses_url = None
    url = 'https://github.com/j-schneider/fly_eye'
    cite = 'schneider2018can'
    animals = ('drosophila')
    real_animals = True    
    year = 2018
    reported_n_total = 2592000
    reported_n_identified = 2592000
    reported_n_photos = 2592000
    reported_n_individuals = 60
    wild = False
    clear_photos = True
    pose = 'single' # from the top
    unique_pattern = True
    from_video = True
    full_frame = False

    def create_catalogue(self):
        data = find_images(self.root)
        folders = data['path'].str.split(os.path.sep, expand=True)

        data['identity'] = 'unknown'
        for i_week in range(1, 4):
            idx1 = folders[0].str.startswith('week' + str(i_week))
            idx2 = folders[1] == 'val'
            idx3 = folders[2].isnull()
            data.loc[idx1 & ~idx2, 'identity'] = (i_week-1)*20 + folders.loc[idx1 & ~idx2, 1].astype(int)
            data.loc[idx1 & idx2 & ~idx3, 'identity'] = (i_week-1)*20 + folders.loc[idx1 & idx2 & ~idx3, 2].astype(int)
            data.loc[idx1 & ~idx2, 'split'] = 'train'
            data.loc[idx1 & idx2, 'split'] = 'val'
        data['id'] = create_id(folders[0] + data['file'])
        data['path'] = data['path'] + os.path.sep + data['file']
        
        df = data.drop(['file'], axis=1)
        return self.finalize_df(df)



class FriesianCattle2015(DatasetFactory):
    licenses = 'Non-Commercial Government Licence for public sector information'
    licenses_url = 'https://www.nationalarchives.gov.uk/doc/non-commercial-government-licence/version/2/'
    url = 'https://data.bris.ac.uk/data/dataset/wurzq71kfm561ljahbwjhx9n3'
    cite = 'andrew2016automatic'
    animals = ('Friesian cattle')
    real_animals = True    
    year = 2016
    reported_n_total = 83+294 # train+test
    reported_n_identified = 83+294 # train+test
    reported_n_photos = 83+294 # train+test
    reported_n_individuals = 40
    wild = False
    clear_photos = True
    pose = 'single' # from the top
    unique_pattern = True
    from_video = True
    full_frame = False

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



class FriesianCattle2017(DatasetFactory):
    licenses = 'Non-Commercial Government Licence for public sector information'
    licenses_url = 'https://www.nationalarchives.gov.uk/doc/non-commercial-government-licence/version/2/'
    url = 'https://data.bris.ac.uk/data/dataset/2yizcfbkuv4352pzc32n54371r'
    cite = 'andrew2017visual'
    animals = ('Friesian cattle')
    real_animals = True    
    year = 2017
    reported_n_total = 940
    reported_n_identified = 940
    reported_n_photos = 940
    reported_n_individuals = 89
    wild = False
    clear_photos = True
    pose = 'single' # from the top
    unique_pattern = True
    from_video = True
    full_frame = False

    def create_catalogue(self):
        data = find_images(self.root)
        folders = data['path'].str.split(os.path.sep, expand=True)

        df = {
            'id': create_id(data['file']),
            'path': data['path'] + os.path.sep + data['file'],
            'identity': folders[1].astype(int),
        }
        return self.finalize_df(df)



class GiraffeZebraID(DatasetFactoryWildMe):
    licenses = 'Attribution-NonCommercial-NoDerivs License'
    licenses_url = 'http://creativecommons.org/licenses/by-nc-nd/2.0/'
    url = 'https://lila.science/datasets/great-zebra-giraffe-id'
    cite = 'parham2017animal'
    animals = ('giraffe masai', 'zebra plains')
    real_animals = True    
    year = 2017
    reported_n_total = 639+6286 # giraffes + zebra
    reported_n_identified = 639+6286 # giraffes + zebra
    reported_n_photos = 4948
    reported_n_individuals = 2056 
    wild = True
    clear_photos = True
    pose = 'double' # from either side
    unique_pattern = True
    from_video = False
    full_frame = True
    
    def create_catalogue(self):
        return self.create_catalogue_wildme('gzgc', 2020)



class Giraffes(DatasetFactory):
    licenses = None
    licenses_url = None
    url = 'ftp://pbil.univ-lyon1.fr/pub/datasets/miele2021'
    cite = 'miele2021revisiting'
    animals = ('giraffe')
    real_animals = True    
    year = 2021
    reported_n_total = None
    reported_n_identified = None 
    reported_n_photos = None
    reported_n_individuals = None 
    wild = True
    clear_photos = True
    pose = 'double' # from either side
    unique_pattern = True
    from_video = True # from burst
    full_frame = False

    def create_catalogue(self):
        path = os.path.join('pbil.univ-lyon1.fr', 'pub', 'datasets', 'miele2021')

        data = find_images(os.path.join(self.root, path))
        folders = data['path'].str.split(os.path.sep, expand=True)

        clusters = folders[0] == 'clusters'
        data, folders = data[clusters], folders[clusters]

        df = {    
            'id': create_id(data['file']),
            'path': path + os.path.sep + data['path'] + os.path.sep + data['file'],
            'identity': folders[1],
        }
        return self.finalize_df(df)



class HappyWhale(DatasetFactory):
    licenses = None
    licenses_url = None
    url = 'https://www.kaggle.com/competitions/happy-whale-and-dolphin'
    cite = 'cheeseman2017happywhale'
    animals = ('whale')
    real_animals = True    
    year = 2022
    reported_n_total = None
    reported_n_identified = None
    reported_n_photos = None
    reported_n_individuals = None
    wild = True
    clear_photos = True
    pose = 'multiple'
    unique_pattern = True
    from_video = False
    full_frame = True
    
    def create_catalogue(self):
        data = pd.read_csv(os.path.join(self.root, 'train.csv'))

        df1 = {
            'id': data['image'].str.split('.', expand=True)[0],
            'path': 'train_images' + os.path.sep + data['image'],
            'identity': data['individual_id'],
            'species': data['species'],
            'split': 'train'
            }
        
        test_files = find_images(os.path.join(self.root, 'test_images'))
        test_files = list(test_files['file'])
        test_files = pd.Series(np.sort(test_files))

        df2 = {
            'id': test_files.str.split('.', expand=True)[0],
            'path': 'test_images' + os.path.sep + test_files,
            'identity': 'unknown',
            'species': 'unknown',
            'split': 'test'
            }
        
        df = pd.concat([pd.DataFrame(df1), pd.DataFrame(df2)])    
        return self.finalize_df(df)



class HumpbackWhaleID(DatasetFactory):
    licenses = None
    licenses_url = None
    url = 'https://www.kaggle.com/competitions/humpback-whale-identification'
    cite = 'humpbackwhale'
    animals = ('whale')
    real_animals = True    
    year = 2019
    reported_n_total = None
    reported_n_identified = None
    reported_n_photos = None
    reported_n_individuals = None 
    wild = True
    clear_photos = True
    pose = 'single' # tail fin only
    unique_pattern = True
    from_video = False
    full_frame = False

    def create_catalogue(self):
        data = pd.read_csv(os.path.join(self.root, 'train.csv'))
        data.loc[data['Id'] == 'new_whale', 'Id'] = 'unknown'

        df1 = {
            'id': data['Image'].str.split('.', expand=True)[0],
            'path': 'train' + os.path.sep + data['Image'],
            'identity': data['Id'],
            'split': 'train'
            }
        
        test_files = find_images(os.path.join(self.root, 'test'))
        test_files = list(test_files['file'])
        test_files = pd.Series(np.sort(test_files))

        df2 = {
            'id': test_files.str.split('.', expand=True)[0],
            'path': 'test' + os.path.sep + test_files,
            'identity': 'unknown',
            'split': 'test'
            }
        
        df = pd.concat([pd.DataFrame(df1), pd.DataFrame(df2)])    
        return self.finalize_df(df)



class HyenaID2022(DatasetFactoryWildMe):
    licenses = 'Attribution-NonCommercial-NoDerivs License'
    licenses_url = 'http://creativecommons.org/licenses/by-nc-nd/2.0/'
    url = 'https://lila.science/datasets/hyena-id-2022/'
    cite = 'botswana2022'
    animals = ('spotted hyena')
    real_animals = True    
    year = 2022
    reported_n_total = 3129
    reported_n_identified = 3129
    reported_n_photos = 3104 
    reported_n_individuals = 256  
    wild = True
    clear_photos = False # night, blurry, parts missing, ...
    pose = 'multiple'
    unique_pattern = True
    from_video = False
    full_frame = True

    def create_catalogue(self):
        return self.create_catalogue_wildme('hyena', 2022)



class IPanda50(DatasetFactory):
    licenses = None
    licenses_url = None
    url = 'https://github.com/iPandaDateset/iPanda-50'
    cite = 'wang2021giant'
    animals = ('great panda')
    real_animals = True    
    year = 2021
    reported_n_total = 6874
    reported_n_identified = 6874
    reported_n_photos = 6874
    reported_n_individuals = 50
    wild = False # zoos
    clear_photos = True
    pose = 'multiple'
    unique_pattern = True
    from_video = False
    full_frame = False

    def create_catalogue(self):
        data = find_images(self.root)
        folders = data['path'].str.split(os.path.sep, expand=True)

        keypoints = []
        for path in data['path'] + os.path.sep + data['file']:
            path_split = os.path.normpath(path).split(os.path.sep)
            path_json = os.path.join('iPanda50-eyes-labels', path_split[1], os.path.splitext(path_split[2])[0] + '.json')

            keypoints_part = np.full(8, np.nan)        
            if os.path.exists(os.path.join(self.root, path_json)):
                with open(os.path.join(self.root, path_json)) as file:
                    keypoints_file = json.load(file)['shapes']
                    if keypoints_file[0]['label'] == 'right_eye':
                        keypoints_part[0:4] = np.reshape(keypoints_file[0]['points'], 4)
                    if keypoints_file[0]['label'] == 'left_eye':
                        keypoints_part[4:8] = np.reshape(keypoints_file[0]['points'], 4)
                    if len(keypoints_file) == 2 and keypoints_file[1]['label'] == 'right_eye':
                        keypoints_part[0:4] = np.reshape(keypoints_file[1]['points'], 4)
                    if len(keypoints_file) == 2 and keypoints_file[1]['label'] == 'left_eye':
                        keypoints_part[4:8] = np.reshape(keypoints_file[1]['points'], 4)
            keypoints.append(list(keypoints_part))
        
        df = {
            'id': create_id(data['file']),
            'path': data['path'] + os.path.sep + data['file'],
            'identity': folders[1],
            'keypoints': keypoints
            }
        return self.finalize_df(df)



class LeopardID2022(DatasetFactoryWildMe):
    licenses = 'Attribution-NonCommercial-NoDerivs License'
    licenses_url = 'http://creativecommons.org/licenses/by-nc-nd/2.0/'
    url = 'https://lila.science/datasets/leopard-id-2022/'
    cite = 'botswana2022'
    animals = ('leopard')
    real_animals = True    
    year = 2022
    reported_n_total = None
    reported_n_identified = 6805
    reported_n_photos = 6795
    reported_n_individuals = 430
    wild = True
    clear_photos = False # night, blurry, parts missing, ...
    pose = 'multiple'
    unique_pattern = True
    from_video = False
    full_frame = True

    def create_catalogue(self):
        return self.create_catalogue_wildme('leopard', 2022)



class LionData(DatasetFactory):
    licenses = None
    licenses_url = None
    url = 'https://github.com/tvanzyl/wildlife_reidentification'
    cite = 'dlamini2020automated'
    animals = ('lion')
    real_animals = True    
    year = 2020
    reported_n_total = 750
    reported_n_identified = 750
    reported_n_photos = 750
    reported_n_individuals = 98
    wild = True
    clear_photos = True
    pose = 'multiple' # various body parts
    unique_pattern = True # by whiskers
    from_video = False
    full_frame = False

    def create_catalogue(self):
        data = find_images(self.root)
        folders = data['path'].str.split(os.path.sep, expand=True)

        identity = folders[3]

        df = {
            'id': create_id(data['file']),
            'path': data['path'] + os.path.sep + data['file'],
            'identity': identity,
        }
        return self.finalize_df(df)



class MacaqueFaces(DatasetFactory):
    licenses = 'The 3-Clause BSD License'
    licenses_url = 'https://github.com/clwitham/MacaqueFaces/blob/master/license.md'
    url = 'https://github.com/clwitham/MacaqueFaces'
    cite = 'witham2018automated'
    animals = ('rhesus macaque')
    real_animals = True    
    year = 2018
    reported_n_total = (150+4*10)*34 # slightly less, not described properly
    reported_n_identified = (150+4*10)*34 # slightly less, not described properly
    reported_n_photos = (150+4*10)*34 # slightly less, not described properly
    reported_n_individuals = 34
    wild = False # breeding facility
    clear_photos = True
    pose = 'single' # from the front
    unique_pattern = False
    from_video = True
    full_frame = False
    
    def create_catalogue(self):
        data = pd.read_csv(os.path.join(self.root, 'MacaqueFaces_ImageInfo.csv'))
        attributes = data[['Category']].to_dict(orient='index')
        
        df = {
            'id': pd.Series(range(len(data))),
            'path': 'MacaqueFaces' + os.path.sep + data['Path'].str.strip(os.path.sep) + os.path.sep + data['FileName'],
            'identity': data['ID'],
            'attributes': attributes,
        }
        return self.finalize_df(df)



class NDD20(DatasetFactory):
    licenses = 'Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)'
    licenses_url = 'https://creativecommons.org/licenses/by-nc-sa/4.0/'
    url = 'https://doi.org/10.25405/data.ncl.c.4982342'
    cite = 'trotter2020ndd20'
    animals = ('Northumberland dolphin')
    real_animals = True    
    year = 2020
    reported_n_total = None # not specified exactly
    reported_n_identified = None # above around 14%
    reported_n_photos = 2201+2201 # above + below
    reported_n_individuals = 82
    wild = False # below the water
    clear_photos = True
    pose = 'multiple' # both above and beow water
    unique_pattern = True
    from_video = False
    full_frame = True

    def create_catalogue(self):
        with open(os.path.join(self.root, 'ABOVE_LABELS.json')) as file:
            data = json.load(file)
        
        entries = []
        for key in data.keys():
            regions = data[key]['regions']
            for region in regions:
                if 'id' in region['region_attributes']:
                    identity = region['region_attributes']['id']
                else:
                    identity = 'unknown'
                segmentation = np.zeros(2*len(region['shape_attributes']['all_points_x']))
                segmentation[0::2] = region['shape_attributes']['all_points_x']
                segmentation[1::2] = region['shape_attributes']['all_points_y']
                entries.append({
                    'identity': identity,
                    'species': region['region_attributes']['species'],
                    'attributes': {'out of focus': np.nan},
                    'file_name': data[key]['filename'],
                    'reg_type': region['shape_attributes']['name'],
                    'segmentation': segmentation,
                    'position': 'above'
                })
        
        with open(os.path.join(self.root, 'BELOW_LABELS.json')) as file:
            data = json.load(file)
            
        for key in data.keys():
            regions = data[key]['regions']
            for region in regions:
                if 'id' in region['region_attributes']:
                    identity = region['region_attributes']['id']
                else:
                    identity = 'unknown'
                segmentation = np.zeros(2*len(region['shape_attributes']['all_points_x']))
                segmentation[0::2] = region['shape_attributes']['all_points_x']
                segmentation[1::2] = region['shape_attributes']['all_points_y']
                entries.append({
                    'identity': identity,
                    'species': np.nan,
                    'attributes': {'out of focus': region['region_attributes']['out of focus']},
                    'file_name': data[key]['filename'],
                    'reg_type': region['shape_attributes']['name'],
                    'segmentation': segmentation,
                    'position': 'below'
                })

        df = pd.DataFrame(entries)
        if len(df.reg_type.unique()) != 1:
            raise(Exception('Multiple segmentation types'))

        df['id'] = range(len(df))
        df['path'] = df['position'].str.upper() + os.path.sep + df['file_name']

        df = df.drop(['reg_type', 'file_name'], axis=1)
        return self.finalize_df(df)



class NOAARightWhale(DatasetFactory):
    licenses = None
    licenses_url = None
    url = 'https://www.kaggle.com/c/noaa-right-whale-recognition'
    cite = 'rightwhale'
    animals = ('right whale')
    real_animals = True    
    year = 2015
    reported_n_total = None
    reported_n_identified = None
    reported_n_photos = None
    reported_n_individuals = None 
    wild = True
    clear_photos = False # often below the water
    pose = 'single' # from the top
    unique_pattern = False # fins not present at pictures
    from_video = False
    full_frame = True

    def create_catalogue(self):
        data = pd.read_csv(os.path.join(self.root, 'train.csv'))
        df1 = {
            #.str.strip('Cow').astype(int)
            'id': data['Image'].str.split('.', expand=True)[0].str.strip('w_').astype(int),
            'path': 'imgs' + os.path.sep + data['Image'],
            'identity': data['whaleID'],
            }

        data = pd.read_csv(os.path.join(self.root, 'sample_submission.csv'))
        df2 = {
            'id': data['Image'].str.split('.', expand=True)[0].str.strip('w_').astype(int),
            'path': 'imgs' + os.path.sep + data['Image'],
            'identity': 'unknown',
            }
        
        df = pd.concat([pd.DataFrame(df1), pd.DataFrame(df2)])    
        return self.finalize_df(df)



class NyalaData(DatasetFactory):
    licenses = None
    licenses_url = None
    url = 'https://github.com/tvanzyl/wildlife_reidentification'
    cite = 'dlamini2020automated'
    animals = ('nyala')
    real_animals = True    
    year = 2020
    reported_n_total = 1934
    reported_n_identified = 1934
    reported_n_photos = 1934
    reported_n_individuals = 274
    wild = True
    clear_photos = True
    pose = 'double' # from either side
    unique_pattern = True
    from_video = False
    full_frame = True

    def create_catalogue(self):
        data = find_images(self.root)
        folders = data['path'].str.split(os.path.sep, expand=True)
        identity = folders[3].astype(int)
        position = np.full(len(data), np.nan, dtype=object)
        position[['left' in filename for filename in data['file']]] = 'left'
        position[['right' in filename for filename in data['file']]] = 'right'

        df = {
            'id': create_id(data['file']),
            'path': data['path'] + os.path.sep + data['file'],
            'identity': identity,
            'position': position,
        }
        return self.finalize_df(df)   



class OpenCows2020(DatasetFactory):
    licenses = 'Non-Commercial Government Licence for public sector information'
    licenses_url = 'https://www.nationalarchives.gov.uk/doc/non-commercial-government-licence/version/2/'
    url = 'https://data.bris.ac.uk/data/dataset/10m32xl88x2b61zlkkgz3fml17'
    cite = 'andrew2021visual'
    animals = ('cow')
    real_animals = True    
    year = 2020
    reported_n_total = 4736
    reported_n_identified = 4736
    reported_n_photos = 4736
    reported_n_individuals = 46
    wild = False
    clear_photos = True
    pose = 'single' # from the top
    unique_pattern = True
    from_video = False
    full_frame = False

    def create_catalogue(self):
        data = find_images(self.root)
        folders = data['path'].str.split(os.path.sep, expand=True)

        #Select only re-identification dataset
        reid = folders[1] == 'identification'
        folders, data = folders[reid], data[reid]

        split = folders[3]
        assert len(split.unique()) == 2
        identity = folders[4]

        df = {
            'id': create_id(identity.astype(str) + split + data['file']),
            'path': data['path'] + os.path.sep + data['file'],
            'identity': identity,
            'split': split
        }
        return self.finalize_df(df)    



class SealID(DatasetFactory):
    licenses = 'Attribution 4.0 International (CC BY 4.0)'
    licenses_url = 'https://creativecommons.org/licenses/by/4.0/'
    url = 'https://doi.org/10.23729/0f4a3296-3b10-40c8-9ad3-0cf00a5a4a53'
    cite = 'nepovinnykh2022sealid'
    animals = ('ringed seal')
    real_animals = True    
    year = 2022
    reported_n_total = 2080
    reported_n_identified = 2080
    reported_n_photos = 2080
    reported_n_individuals = 57
    wild = True
    clear_photos = False # night, blurry, parts missing, ...
    pose = 'multiple'
    unique_pattern = True 
    from_video = False
    full_frame = True

    def create_catalogue(self, variant='source'):
        if variant == 'source':
            prefix = 'source_'
        elif variant == 'segmented':
            prefix = 'segmented_'
        else:
            raise ValueError(f'Variant {variant} is not valid')

        data = pd.read_csv(os.path.join(self.root, 'full images', 'annotation.csv'))

        df = {    
            'id': data['file'].str.split('.', expand=True)[0],
            'path': 'full images' + os.path.sep + prefix + data['reid_split'] + os.path.sep + data['file'],
            'identity': data['class_id'].astype(int),
            'reid_split': data['reid_split'],
            'segmentation_split': data['segmentation_split'],
        }
        return self.finalize_df(df)



class SMALST(DatasetFactory):
    licenses = 'MIT License'
    licenses_url = 'https://github.com/silviazuffi/smalst/blob/master/LICENSE.txt'
    url = 'https://github.com/silviazuffi/smalst'
    cite = 'zuffi2019three'
    animals = ('zebra')
    real_animals = False
    year = 2019
    reported_n_total = 12850
    reported_n_identified = 12850
    reported_n_photos = 12850
    reported_n_individuals = 10
    wild = False
    clear_photos = True
    pose = 'multiple'
    unique_pattern = True 
    from_video = False
    full_frame = True

    def create_catalogue(self):
        data = find_images(os.path.join(self.root, 'zebra_training_set', 'images'))

        path0 = data['file'].str.strip('zebra_')
        data['identity'] = path0.str[0]
        data['id'] = [int(p[1:].strip('_frame_').split('.')[0]) for p in path0]
        data['path'] = 'zebra_training_set' + os.path.sep + 'images' + os.path.sep + data['file']
        data = data.drop(['file'], axis=1)

        masks = find_images(os.path.join(self.root, 'zebra_training_set', 'bgsub'))

        path0 = masks['file'].str.strip('zebra_')
        masks['id'] = [int(p[1:].strip('_frame_').split('.')[0]) for p in path0]
        masks['mask'] = 'zebra_training_set' + os.path.sep + 'bgsub' + os.path.sep + masks['file']
        masks = masks.drop(['path', 'file'], axis=1)

        df = pd.merge(data, masks, on='id')
        return self.finalize_df(df)



class StripeSpotter(DatasetFactory):
    licenses = 'GNU General Public License, version 2'
    licenses_url = 'http://www.gnu.org/licenses/old-licenses/gpl-2.0.html'
    url = 'https://code.google.com/archive/p/stripespotter/downloads'
    cite = 'lahiri2011biometric'
    animals = ('zebra')
    real_animals = True    
    year = 2011
    reported_n_total = None
    reported_n_identified = None
    reported_n_photos = None
    reported_n_individuals = None
    wild = True
    clear_photos = True
    pose = 'double' # from either side
    unique_pattern = True
    from_video = False
    full_frame = True

    def create_catalogue(self):
        data = find_images(self.root)
        data['index'] = data['file'].str[-7:-4].astype(int)
        category = data['file'].str.split('-', expand=True)[0]
        data = data[category == 'img'] # Only full images
        
        data_aux = pd.read_csv(os.path.join(self.root, 'data', 'SightingData.csv'))
        data = pd.merge(data, data_aux, how='left', left_on='index', right_on='#imgindex')
        data.loc[data['animal_name'].isnull(), 'animal_name'] = 'unknown'
        attributes = data[['flank', 'photo_quality']].to_dict(orient='index')

        df = {
            'id': create_id(data['file']),
            'path':  data['path'] + os.path.sep + data['file'],
            'identity': data['animal_name'],
            'bbox': pd.Series([[int(a) for a in b.split(' ')] for b in data['roi']]),
            'attributes': attributes,
        }
        return self.finalize_df(df)  



class WhaleSharkID(DatasetFactoryWildMe):
    licenses = 'Attribution-NonCommercial-NoDerivs 2.0 Generic (CC BY-NC-ND 2.0)'
    licenses_url = 'http://creativecommons.org/licenses/by-nc-nd/2.0/'
    url = 'https://lila.science/datasets/whale-shark-id'
    cite = 'holmberg2009estimating'
    animals = ('whale shark')
    real_animals = True    
    year = 2020
    reported_n_total = 7693
    reported_n_identified = 7693
    reported_n_photos = 7693
    reported_n_individuals = 543
    wild = True
    clear_photos = False
    pose = 'multiple' # underwater
    unique_pattern = True
    from_video = False
    full_frame = True

    def create_catalogue(self):
        return self.create_catalogue_wildme('whaleshark', 2020)



class WNIGiraffes(DatasetFactory):
    licenses = 'Community Data License Agreement – Permissive'
    licenses_url = 'https://cdla.dev/permissive-1-0/'
    url = 'https://lila.science/datasets/wni-giraffes'
    cite = 'wnigiraffes'
    animals = ('giraffe')
    real_animals = True    
    year = 2021
    reported_n_total = 29806
    reported_n_identified = 29806
    reported_n_photos = 29806 
    reported_n_individuals = None
    wild = True
    clear_photos = True
    pose = 'double' # from either side
    unique_pattern = True
    from_video = False
    full_frame = True

    def create_catalogue(self):
        data = find_images(self.root)
        folders = data['path'].str.split(os.path.sep, expand=True)
        
        with open(os.path.join(self.root, 'wni_giraffes_train.json')) as file:
            keypoints = json.load(file)
        create_dict = lambda i: {'file': os.path.split(i['filename'])[1], 'keypoints': self.extract_keypoints(i)}
        df_keypoints = pd.DataFrame([create_dict(i) for i in keypoints['annotations']])

        data = pd.merge(data, df_keypoints, how='left', on='file')
        data['id'] = create_id(data['file'])
        data['identity'] = folders[1].astype(int)
        data['path'] = data['path'] + os.path.sep + data['file']
        data = data.drop(['file'], axis=1)

        return self.finalize_df(data)

    def extract_keypoints(self, row):
        keypoints = [row['keypoints']['too']['median_x'], row['keypoints']['too']['median_y'],
                row['keypoints']['toh']['median_x'], row['keypoints']['toh']['median_y'],
                row['keypoints']['ni']['median_x'], row['keypoints']['ni']['median_y'],
                row['keypoints']['fbh']['median_x'], row['keypoints']['fbh']['median_y'],
            ]
        keypoints = np.array(keypoints)
        keypoints[keypoints == None] = np.nan
        return list(keypoints)



class ZindiTurtleRecall(DatasetFactory):
    licenses = 'Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)'
    licenses_url = 'https://creativecommons.org/licenses/by-sa/4.0/'
    url = 'https://zindi.africa/competitions/turtle-recall-conservation-challenge'
    cite = 'zinditurtles'
    animals = ('sea turtle')
    real_animals = True    
    year = 2022
    reported_n_total = None
    reported_n_identified = None
    reported_n_photos = None
    reported_n_individuals = None
    wild = True
    clear_photos = True
    pose = 'double' # from either side
    unique_pattern = True
    from_video = False
    full_frame = False

    def create_catalogue(self):
        data_train = pd.read_csv(os.path.join(self.root, 'train.csv'))
        data_train['split'] = 'train'
        data_test = pd.read_csv(os.path.join(self.root, 'test.csv'))
        data_test['split'] = 'test'
        data_extra = pd.read_csv(os.path.join(self.root, 'extra_images.csv'))
        data_extra['split'] = 'unassigned'
        data = pd.concat([data_train, data_test, data_extra])

        data = data.reset_index(drop=True)
        
        data.loc[data['turtle_id'].isnull(), 'turtle_id'] = 'unknown'
        df = {
            'id': data['image_id'],
            'path': 'images' + os.path.sep + data['image_id'] + '.JPG',
            'identity': data['turtle_id'],
            'position': data['image_location'].str.lower(),
            'split': data['split'],
        }
        return self.finalize_df(df)



