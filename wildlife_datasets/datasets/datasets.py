import os
import pandas as pd
import numpy as np
from typing import Tuple, Optional
import hashlib
import json
import datetime
from collections.abc import Iterable

from .. import downloads
from .. import utils
from .metadata import metadata


'''
General:

TODO: I would represent keypoints as they are. There is no unified notation
what they can represent (unlike segmentations and bbox) and lot of
imporant information can get lost.

TODO: We should at least provide description on how we did the data
processing and the reasoning behind it. Some datasets are especially
dificult and not obvious

'''

def find_images(
    root: str,
    img_extensions: Tuple[str, ...] = ('.png', '.jpg', '.jpeg')
    ) -> pd.DataFrame:
    '''
    Find all image files in folder recursively based on img_extensions. 
    Save filename and relative path from root.
    '''
    data = [] 
    for path, directories, files in os.walk(root):
        for file in files:
            if file.lower().endswith(tuple(img_extensions)):
                data.append({'path': os.path.relpath(path, start=root), 'file': file})
    return pd.DataFrame(data)

def create_id(string_col: pd.Series) -> pd.Series:
    '''
    Creates unique id from string based on MD5 hash.
    '''
    entity_id = string_col.apply(lambda x: hashlib.md5(x.encode()).hexdigest()[:16])
    assert len(entity_id.unique()) == len(entity_id)
    return entity_id

def convert_keypoint(keypoint, keypoints_names):
    keypoint_dict = {}
    if isinstance(keypoint, Iterable):
        for i in range(len(keypoints_names)):
            x = keypoint[2*i]
            y = keypoint[2*i+1]
            if np.isfinite(x) and np.isfinite(y):
                keypoint_dict[keypoints_names[i]] = [x, y]
    return keypoint_dict

def convert_keypoints(keypoints: pd.Series, keypoints_names):
    return [convert_keypoint(keypoint, keypoints_names) for keypoint in keypoints]


class DatasetFactory():
    def __init__(
        self, 
        root: str,
        df: Optional[pd.DataFrame] = None,
        download: bool = False,
        **kwargs
        ):

        self.root = root
        if download and hasattr(self, 'download'): 
            self.download.get_data(root)
        if df is None:
            self.df = self.create_catalogue(**kwargs)
        else:
            self.df = df

    def create_catalogue(self):
        '''
        Create catalogue data frame.
        This method is dataset specific and each dataset needs to override it.
        '''
        raise NotImplementedError()

    def finalize_catalogue(self, df: pd.DataFrame) -> pd.DataFrame:
        '''
        Finalize catalogue data frame and runs checks.
        '''
        df = self.reorder_df(df)
        df = self.remove_constant_columns(df)
        self.check_unique_id(df)
        self.check_files_exist(df['path'])
        if 'segmentation' in df.columns:
            self.check_files_exist(df['segmentation'])
        return df

    def reorder_df(self, df: pd.DataFrame) -> pd.DataFrame:
        default_order = ['id', 'identity', 'path', 'bbox', 'segmentation', 'position', 'species', 'keypoints']
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

    def remove_constant_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        '''
        Removes columns with single unique value.
        '''
        for df_name in list(df.columns):
            if df[df_name].astype('str').nunique() == 1:
                df = df.drop([df_name], axis=1)
        return df

    def check_unique_id(self, df: pd.DataFrame) -> None:
        '''
        Check if values in ID column are unique.
        '''
        if len(df['id'].unique()) != len(df):
            raise(Exception('Image ID not unique.'))

    def check_files_exist(self, col: pd.Series) -> None:
        '''
        Check if paths in given column exist.
        '''
        for path in col:
            if type(path) == str and not os.path.exists(os.path.join(self.root, path)):
                raise(Exception('Path does not exist:' + os.path.join(self.root, path)))


class Test(DatasetFactory):
    download = downloads.test
    metadata = metadata['Test']

    def create_catalogue(self) -> pd.DataFrame:
        return pd.DataFrame([1, 2])


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
            
        create_dict = lambda i: {'identity': i['name'], 'bbox': utils.analysis.segmentation_bbox(i['segmentation'][0]), 'image_id': i['image_id'], 'category_id': i['category_id'], 'segmentation': i['segmentation'][0], 'position': i['viewpoint']}
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
            ii.append(utils.analysis.is_annotation_bbox(df.iloc[i]['segmentation'], df.iloc[i]['bbox'], tol=3))
        df.loc[ii, 'segmentation'] = np.nan

        # Rename empty dates
        df.loc[df['date'] == 'NA', 'date'] = np.nan

        df = df.drop(['image_id', 'file_name', 'supercategory', 'category_id'], axis=1)
        if len(df['species'].unique()) == 1:
            df = df.drop(['species'], axis=1)
        return self.finalize_catalogue(df)



class AAUZebraFishID(DatasetFactory):
    download = downloads.aau_zebrafish_id
    metadata = metadata['AAUZebraFishID']
    
    def create_catalogue(self) -> pd.DataFrame:
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
        attributes.drop([0], axis=1, inplace=True)
        attributes.columns = ['turning', 'occlusion', 'glitch']
        attributes = attributes.astype(int).astype(bool)

        position = data['Right,Turning,Occlusion,Glitch'].str.split(',', expand=True)[0]
        position.replace('1', 'right', inplace=True)
        position.replace('0', 'left', inplace=True)

        video = data['Filename'].str.split('_',  expand=True)[0]
        video = video.astype('category').cat.codes

        df = pd.DataFrame({
            'id': create_id(data['Object ID'].astype(str) + data['Filename']),
            'path': 'data' + os.path.sep + data['Filename'],
            'identity': data['Object ID'],
            'video': video,
            'bbox': bbox,
            'position': position,
        })
        df = df.join(attributes)
        return self.finalize_catalogue(df)


class AerialCattle2017(DatasetFactory):
    download = downloads.aerial_cattle_2017
    metadata = metadata['AerialCattle2017']

    def create_catalogue(self) -> pd.DataFrame:
        data = find_images(self.root)
        folders = data['path'].str.split(os.path.sep, expand=True)

        df = pd.DataFrame({
            'id': create_id(data['file']),
            'path': data['path'] + os.path.sep + data['file'],
            'identity': folders[1].astype(int),
            'video': folders[2],
        })
        return self.finalize_catalogue(df)


class ATRW(DatasetFactory):
    download = downloads.atrw
    metadata = metadata['ATRW']

    def create_catalogue(self) -> pd.DataFrame:
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
        return self.finalize_catalogue(df)

    
    
class BelugaID(DatasetFactoryWildMe):
    download = downloads.beluga_id
    metadata = metadata['BelugaID']

    def create_catalogue(self) -> pd.DataFrame:
        return self.create_catalogue_wildme('beluga', 2022)



class BirdIndividualID(DatasetFactory):
    download = downloads.bird_individual_id
    metadata = metadata['BirdIndividualID']

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

        df1 = pd.DataFrame({    
            'id': create_id(split + data['file']),
            'path': prefix1 + os.path.sep + prefix2 + os.path.sep + data['path'] + os.path.sep + data['file'],
            'identity': identity,
            'species': species,
            'split': split,
        })

        # Add images without labels
        path = os.path.join(self.root, prefix1, 'New_birds')
        data = find_images(path)
        species = data['path']

        df2 = pd.DataFrame({    
            'id': create_id(data['file']),
            'path': prefix1 + os.path.sep + 'New_birds' + os.path.sep + data['path'] + os.path.sep + data['file'],
            'identity': 'unknown',
            'species': species,
            'split': 'unassigned',
        })

        return self.finalize_catalogue(pd.concat([df1, df2]))


class CTai(DatasetFactory):
    download = downloads.c_tai
    metadata = metadata['CTai']

    def create_catalogue(self) -> pd.DataFrame:
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
        
        keypoints = data[[11, 12, 14, 15, 17, 18, 20, 21, 23, 24]].to_numpy()
        keypoints[np.isinf(keypoints)] = np.nan
        keypoints = pd.Series(list(keypoints))
        
        df = pd.DataFrame({
            'id': pd.Series(range(len(data))),
            'path': path + os.path.sep + data[1],
            'identity': data[3],
            'keypoints': keypoints,
            'age': data[5],
            'age_group': data[7],
            'gender': data[9],
        })
        for replace_tuple in replace_names:
            df['identity'] = df['identity'].replace({replace_tuple[0]: replace_tuple[1]})
        return self.finalize_catalogue(df)



class CZoo(DatasetFactory):
    download = downloads.c_zoo
    metadata = metadata['CZoo']

    def create_catalogue(self) -> pd.DataFrame:
        path = os.path.join('chimpanzee_faces-master', 'datasets_cropped_chimpanzee_faces', 'data_CZoo',)
        data = pd.read_csv(os.path.join(self.root, path, 'annotations_czoo.txt'), header=None, sep=' ')

        keypoints = data[[11, 12, 14, 15, 17, 18, 20, 21, 23, 24]].to_numpy()
        keypoints[np.isinf(keypoints)] = np.nan
        keypoints = pd.Series(list(keypoints))
        
        df = pd.DataFrame({
            'id': pd.Series(range(len(data))),
            'path': path + os.path.sep + data[1],
            'identity': data[3],
            'keypoints': keypoints,
            'age': data[5],
            'age_group': data[7],
            'gender': data[9],
        })
        return self.finalize_catalogue(df)



class Cows2021(DatasetFactory):
    download = downloads.cows_2021
    metadata = metadata['Cows2021']

    def create_catalogue(self) -> pd.DataFrame:
        data = find_images(self.root)
        folders = data['path'].str.split(os.path.sep, expand=True)

        ii = (folders[2] == 'Identification') & (folders[3] == 'Test')
        folders = folders.loc[ii]
        data = data.loc[ii]

        df = pd.DataFrame({
            'id': create_id(data['file']),
            'path': data['path'] + os.path.sep + data['file'],
            'identity': folders[4].astype(int),
        })
        return self.finalize_catalogue(df)



class Drosophila(DatasetFactory):
    download = downloads.drosophila
    metadata = metadata['Drosophila']

    def create_catalogue(self) -> pd.DataFrame:
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
        return self.finalize_catalogue(df)



class FriesianCattle2015(DatasetFactory):
    download = downloads.friesian_cattle_2015
    metadata = metadata['FriesianCattle2015']

    def create_catalogue(self) -> pd.DataFrame:
        data = find_images(self.root)
        folders = data['path'].str.split(os.path.sep, expand=True)
        split = folders[1].replace({'Cows-testing': 'test', 'Cows-training': 'train'})
        assert len(split.unique()) == 2

        identity = folders[2].str.strip('Cow').astype(int)

        df = pd.DataFrame({
            'id': create_id(identity.astype(str) + split + data['file']),
            'path': data['path'] + os.path.sep + data['file'],
            'identity': identity,
            'split': split
        })
        return self.finalize_catalogue(df)



class FriesianCattle2017(DatasetFactory):
    download = downloads.friesian_cattle_2017
    metadata = metadata['FriesianCattle2017']

    def create_catalogue(self) -> pd.DataFrame:
        data = find_images(self.root)
        folders = data['path'].str.split(os.path.sep, expand=True)

        df = pd.DataFrame({
            'id': create_id(data['file']),
            'path': data['path'] + os.path.sep + data['file'],
            'identity': folders[1].astype(int),
        })
        return self.finalize_catalogue(df)


class GiraffeZebraID(DatasetFactoryWildMe):
    download = downloads.giraffe_zebra_id
    metadata = metadata['GiraffeZebraID']
    
    def create_catalogue(self) -> pd.DataFrame:
        return self.create_catalogue_wildme('gzgc', 2020)



class Giraffes(DatasetFactory):
    download = downloads.giraffes
    metadata = metadata['Giraffes']

    def create_catalogue(self) -> pd.DataFrame:
        path = os.path.join('pbil.univ-lyon1.fr', 'pub', 'datasets', 'miele2021')

        data = find_images(os.path.join(self.root, path))
        folders = data['path'].str.split(os.path.sep, expand=True)

        clusters = folders[0] == 'clusters'
        data, folders = data[clusters], folders[clusters]

        df = pd.DataFrame({    
            'id': create_id(data['file']),
            'path': path + os.path.sep + data['path'] + os.path.sep + data['file'],
            'identity': folders[1],
        })
        return self.finalize_catalogue(df)



class HappyWhale(DatasetFactory):
    download = downloads.happy_whale
    metadata = metadata['HappyWhale']
    
    def create_catalogue(self) -> pd.DataFrame:
        replace_names = [
            ('bottlenose_dolpin', 'bottlenose_dolphin'),
            ('kiler_whale', 'killer_whale'),
        ]

        data = pd.read_csv(os.path.join(self.root, 'train.csv'))

        df1 = pd.DataFrame({
            'id': data['image'].str.split('.', expand=True)[0],
            'path': 'train_images' + os.path.sep + data['image'],
            'identity': data['individual_id'],
            'species': data['species'],
            'split': 'train'
            })
        for replace_tuple in replace_names:
            df1['species'] = df1['species'].replace({replace_tuple[0]: replace_tuple[1]})

        test_files = find_images(os.path.join(self.root, 'test_images'))
        test_files = list(test_files['file'])
        test_files = pd.Series(np.sort(test_files))

        df2 = pd.DataFrame({
            'id': test_files.str.split('.', expand=True)[0],
            'path': 'test_images' + os.path.sep + test_files,
            'identity': 'unknown',
            'species': np.nan,
            'split': 'test'
            })
        
        df = pd.concat([df1, df2])    
        return self.finalize_catalogue(df)



class HumpbackWhaleID(DatasetFactory):
    download = downloads.humpback_whale
    metadata = metadata['HumpbackWhaleID']

    def create_catalogue(self) -> pd.DataFrame:
        data = pd.read_csv(os.path.join(self.root, 'train.csv'))
        data.loc[data['Id'] == 'new_whale', 'Id'] = 'unknown'

        df1 = pd.DataFrame({
            'id': data['Image'].str.split('.', expand=True)[0],
            'path': 'train' + os.path.sep + data['Image'],
            'identity': data['Id'],
            'split': 'train'
            })
        
        test_files = find_images(os.path.join(self.root, 'test'))
        test_files = list(test_files['file'])
        test_files = pd.Series(np.sort(test_files))

        df2 = pd.DataFrame({
            'id': test_files.str.split('.', expand=True)[0],
            'path': 'test' + os.path.sep + test_files,
            'identity': 'unknown',
            'split': 'test'
            })
        
        df = pd.concat([df1, df2])    
        return self.finalize_catalogue(df)



class HyenaID2022(DatasetFactoryWildMe):
    download = downloads.hyena_id_2022
    metadata = metadata['HyenaID2022']

    def create_catalogue(self) -> pd.DataFrame:
        return self.create_catalogue_wildme('hyena', 2022)



class IPanda50(DatasetFactory):
    download = downloads.ipanda_50
    metadata = metadata['IPanda50']

    def create_catalogue(self) -> pd.DataFrame:
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
        
        df = pd.DataFrame({
            'id': create_id(data['file']),
            'path': data['path'] + os.path.sep + data['file'],
            'identity': folders[1],
            'keypoints': keypoints
            })
        return self.finalize_catalogue(df)



class LeopardID2022(DatasetFactoryWildMe):
    download = downloads.leopard_id_2022
    metadata = metadata['LeopardID2022']

    def create_catalogue(self) -> pd.DataFrame:
        return self.create_catalogue_wildme('leopard', 2022)



class LionData(DatasetFactory):
    download = downloads.lion_data
    metadata = metadata['LionData']

    def create_catalogue(self) -> pd.DataFrame:
        data = find_images(self.root)
        folders = data['path'].str.split(os.path.sep, expand=True)

        df = pd.DataFrame({
            'id': create_id(data['file']),
            'path': data['path'] + os.path.sep + data['file'],
            'identity': folders[3],
        })
        return self.finalize_catalogue(df)



class MacaqueFaces(DatasetFactory):
    download = downloads.macaque_faces
    metadata = metadata['MacaqueFaces']
    
    def create_catalogue(self) -> pd.DataFrame:
        data = pd.read_csv(os.path.join(self.root, 'MacaqueFaces_ImageInfo.csv'))
        date_taken = [datetime.datetime.strptime(date, '%d-%m-%Y').strftime('%Y-%m-%d') for date in data['DateTaken']]
        
        df = pd.DataFrame({
            'id': pd.Series(range(len(data))),
            'path': 'MacaqueFaces' + os.path.sep + data['Path'].str.strip(os.path.sep) + os.path.sep + data['FileName'],
            'identity': data['ID'],
            'date': pd.Series(date_taken),
            'category': data['Category']
        })
        return self.finalize_catalogue(df)



class NDD20(DatasetFactory):
    download = downloads.ndd20
    metadata = metadata['NDD20']

    def create_catalogue(self) -> pd.DataFrame:
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
                    'out_of_focus': np.nan,
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
                    'species': 'WBD',
                    'out_of_focus': region['region_attributes']['out of focus'] == 'true',
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
        return self.finalize_catalogue(df)



class NOAARightWhale(DatasetFactory):
    download = downloads.noaa_right_whale
    metadata = metadata['NOAARightWhale']

    def create_catalogue(self) -> pd.DataFrame:
        data = pd.read_csv(os.path.join(self.root, 'train.csv'))
        df1 = pd.DataFrame({
            #.str.strip('Cow').astype(int)
            'id': data['Image'].str.split('.', expand=True)[0].str.strip('w_').astype(int),
            'path': 'imgs' + os.path.sep + data['Image'],
            'identity': data['whaleID'],
            })

        data = pd.read_csv(os.path.join(self.root, 'sample_submission.csv'))
        df2 = pd.DataFrame({
            'id': data['Image'].str.split('.', expand=True)[0].str.strip('w_').astype(int),
            'path': 'imgs' + os.path.sep + data['Image'],
            'identity': 'unknown',
            })
        
        df = pd.concat([df1, df2])    
        return self.finalize_catalogue(df)



class NyalaData(DatasetFactory):
    download = downloads.nyala_data
    metadata = metadata['NyalaData']

    def create_catalogue(self) -> pd.DataFrame:
        data = find_images(self.root)
        folders = data['path'].str.split(os.path.sep, expand=True)
        identity = folders[3].astype(int)
        position = np.full(len(data), np.nan, dtype=object)
        position[['left' in filename for filename in data['file']]] = 'left'
        position[['right' in filename for filename in data['file']]] = 'right'

        df = pd.DataFrame({
            'id': create_id(data['file']),
            'path': data['path'] + os.path.sep + data['file'],
            'identity': identity,
            'position': position,
        })
        return self.finalize_catalogue(df)   



class OpenCows2020(DatasetFactory):
    download = downloads.open_cows_2020
    metadata = metadata['OpenCows2020']

    def create_catalogue(self) -> pd.DataFrame:
        data = find_images(self.root)
        folders = data['path'].str.split(os.path.sep, expand=True)

        #Select only re-identification dataset
        reid = folders[1] == 'identification'
        folders, data = folders[reid], data[reid]

        split = folders[3]
        assert len(split.unique()) == 2
        identity = folders[4]

        df = pd.DataFrame({
            'id': create_id(identity.astype(str) + split + data['file']),
            'path': data['path'] + os.path.sep + data['file'],
            'identity': identity,
            'split': split
        })
        return self.finalize_catalogue(df)    



class SealID(DatasetFactory):
    download = downloads.seal_id
    metadata = metadata['SealID']

    def create_catalogue(self, variant='source'):
        if variant == 'source':
            prefix = 'source_'
        elif variant == 'segmented':
            prefix = 'segmented_'
        else:
            raise ValueError(f'Variant {variant} is not valid')

        data = pd.read_csv(os.path.join(self.root, 'full images', 'annotation.csv'))

        df = pd.DataFrame({    
            'id': data['file'].str.split('.', expand=True)[0],
            'path': 'full images' + os.path.sep + prefix + data['reid_split'] + os.path.sep + data['file'],
            'identity': data['class_id'].astype(int),
            'reid_split': data['reid_split'],
            'segmentation_split': data['segmentation_split'],
        })
        return self.finalize_catalogue(df)


class SMALST(DatasetFactory):
    download = downloads.smalst
    metadata = metadata['SMALST']

    def create_catalogue(self) -> pd.DataFrame:
        # Images
        data = find_images(os.path.join(self.root, 'zebra_training_set', 'images'))
        path = data['file'].str.strip('zebra_')
        data['identity'] = path.str[0]
        data['id'] = [int(p[1:].strip('_frame_').split('.')[0]) for p in path]
        data['path'] = 'zebra_training_set' + os.path.sep + 'images' + os.path.sep + data['file']
        data = data.drop(['file'], axis=1)

        # Masks
        masks = find_images(os.path.join(self.root, 'zebra_training_set', 'bgsub'))
        path = masks['file'].str.strip('zebra_')
        masks['id'] = [int(p[1:].strip('_frame_').split('.')[0]) for p in path]
        masks['segmentation'] = 'zebra_training_set' + os.path.sep + 'bgsub' + os.path.sep + masks['file']
        masks = masks.drop(['path', 'file'], axis=1)

        df = pd.merge(data, masks, on='id')
        return self.finalize_catalogue(df)


class StripeSpotter(DatasetFactory):
    download = downloads.stripe_spotter
    metadata = metadata['StripeSpotter']

    def create_catalogue(self) -> pd.DataFrame:
        data = find_images(self.root)
        data['index'] = data['file'].str[-7:-4].astype(int)
        category = data['file'].str.split('-', expand=True)[0]
        data = data[category == 'img'] # Only full images
        
        data_aux = pd.read_csv(os.path.join(self.root, 'data', 'SightingData.csv'))
        data = pd.merge(data, data_aux, how='left', left_on='index', right_on='#imgindex')
        data.loc[data['animal_name'].isnull(), 'animal_name'] = 'unknown'
        
        df = pd.DataFrame({
            'id': create_id(data['file']),
            'path':  data['path'] + os.path.sep + data['file'],
            'identity': data['animal_name'],
            'bbox': pd.Series([[int(a) for a in b.split(' ')] for b in data['roi']]),
            'position': data['flank'],
            'photo_quality': data['photo_quality'],
        })
        return self.finalize_catalogue(df)  



class WhaleSharkID(DatasetFactoryWildMe):
    download = downloads.whale_shark_id
    metadata = metadata['WhaleSharkID']

    def create_catalogue(self) -> pd.DataFrame:
        return self.create_catalogue_wildme('whaleshark', 2020)



class WNIGiraffes(DatasetFactory):
    download = downloads.wni_giraffes
    metadata = metadata['WNIGiraffes']

    def create_catalogue(self) -> pd.DataFrame:
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

        return self.finalize_catalogue(data)

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
    download = downloads.zindi_turtle_recall
    metadata = metadata['ZindiTurtleRecall']

    def create_catalogue(self) -> pd.DataFrame:
        data_train = pd.read_csv(os.path.join(self.root, 'train.csv'))
        data_train['split'] = 'train'
        data_test = pd.read_csv(os.path.join(self.root, 'test.csv'))
        data_test['split'] = 'test'
        data_extra = pd.read_csv(os.path.join(self.root, 'extra_images.csv'))
        data_extra['split'] = 'unassigned'
        data = pd.concat([data_train, data_test, data_extra])

        data = data.reset_index(drop=True)
        
        data.loc[data['turtle_id'].isnull(), 'turtle_id'] = 'unknown'
        df = pd.DataFrame({
            'id': data['image_id'],
            'path': 'images' + os.path.sep + data['image_id'] + '.JPG',
            'identity': data['turtle_id'],
            'position': data['image_location'].str.lower(),
            'split': data['split'],
        })
        return self.finalize_catalogue(df)



