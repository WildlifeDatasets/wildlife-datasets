import os
import pandas as pd
from . import utils
from .datasets import WildlifeDataset

summary = {
    'licenses': 'Non-Commercial Government Licence for public sector information',
    'licenses_url': 'https://www.nationalarchives.gov.uk/doc/non-commercial-government-licence/version/2/',
    'url': 'https://data.bris.ac.uk/data/dataset/jf0859kboy8k2ufv60dqeb2t8',
    'publication_url': 'https://arxiv.org/abs/2012.04689',
    'cite': 'brookes2020dataset',
    'animals': {'gorilla'},
    'animals_simple': 'gorillas',
    'real_animals': True,
    'year': 2020,
    'reported_n_total': 5428,
    'reported_n_individuals': 7,
    'wild': False,
    'clear_photos': True,
    'pose': 'single',
    'unique_pattern': False,
    'from_video': True,
    'cropped': False,
    'span': 'short',
    'size': 15600,
}

class BristolGorillas2020(WildlifeDataset):
    summary = summary
    url = 'https://data.bris.ac.uk/datasets/tar/jf0859kboy8k2ufv60dqeb2t8.zip'
    archive = 'jf0859kboy8k2ufv60dqeb2t8.zip'

    @classmethod
    def _download(cls):
        command = f"wget -c -q {cls.url}"
        exception_text = '''Download works only on Linux. Please download it manually.
            Check https://wildlifedatasets.github.io/wildlife-datasets/downloads#bristolgorillas2020'''
        if os.name == 'posix':
            os.system(command)
        else:
            raise Exception(exception_text)

    @classmethod
    def _extract(cls):
        exception_text = '''Extracting resulted in the expection below.
            This may have happened that because the file was not downloaded correctly.
            Download manually or try to call (the download should resume):
            datasets.BristolGorillas2020.get_data(root, force=True)'''
        try:
            utils.extract_archive(cls.archive, delete=True)
        except Exception as e:
            print(e)            
            raise Exception(exception_text)
    
    def create_catalogue(self) -> pd.DataFrame:
        assert self.root is not None
        data = utils.find_images(self.root)
        folders = data['path'].str.split(os.path.sep, expand=True)
        n_folders = max(folders.columns)

        # Restrict to correct images
        idx = folders[n_folders-2] != 'videos'
        data = data[idx]
        folders = folders[idx]

        # Create the dataframe
        df1 = pd.DataFrame({
            'path': data['path'] + os.path.sep + data['file'],
            'original_split': folders[n_folders-2],
        })
        
        # Add bounding boxes (either load them or create and save them)
        identity_conversion = {
            0: 'afia',
            1: 'ayana',
            2: 'jock',
            3: 'kala',
            4: 'kera',
            5: 'kukuena',
            6: 'touni'
        }

        identity_all = []
        bbox_all = []
        path_all = []
        for i in range(len(df1)):
            path_img = os.path.join(self.root, df1['path'].iloc[i])
            path_bbox = os.path.splitext(path_img)[0] + '.txt'
            path_size = os.path.splitext(path_img)[0] + '_size.txt'

            # Load image size
            if os.path.exists(path_size):
                with open(path_size, 'r') as file:
                    line = file.readline()
                w, h = [int(num) for num in line.split()]
            else:
                img = utils.load_image(path_img)
                w, h = img.size
                with open(path_size, 'w') as file:
                    file.write(f'{w} {h}\n')

            # Load bounding boxes
            with open(path_bbox, 'r') as file:
                lines = [line.rstrip() for line in file]
            for line in lines:
                if len(line) > 0:
                    identity = identity_conversion[int(line[0])]
                    bbox = [float(num) for num in line.split()[1:]]
                    bbox = utils.yolo_to_pascalvoc(*bbox, w, h)
                    bbox = [bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1]]
                    identity_all.append(identity)
                    bbox_all.append(bbox)
                    path_all.append(df1['path'].iloc[i])

        # Merge bounding boxes into paths and remove images with no bounding boxes
        df2 = pd.DataFrame({'identity': identity_all, 'bbox': bbox_all, 'path': path_all})
        df = pd.merge(df1, df2, on='path', how='left')
        df = df[~df['identity'].isnull()]

        # Finalize catalogue
        df['image_id'] = utils.create_id(df['path'].apply(lambda x: os.path.basename(x)) + df['identity'])
        return self.finalize_catalogue(df)
