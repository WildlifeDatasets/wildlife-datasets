import os
import json
import ast

import numpy as np
import pandas as pd
from .downloads import DownloadINaturalist
from .datasets import WildlifeDataset

summary = {
    "licenses": "",
    "licenses_url": "",
    "url": "",
    "publication_url": None,
    "cite": "",
    "animals": {"Green Sea Turtle"},
    "animals_simple": "sea turtles",
    "real_animals": True,
    "year": 2025,
    "reported_n_total": None,
    "reported_n_individuals": None,
    "wild": True,
    "clear_photos": True,
    "pose": "multiple",
    "unique_pattern": True,
    "from_video": False,
    "cropped": False,
    "span": "",
    "size": None,
}


class TurtlesOfSMSRC(DownloadINaturalist, WildlifeDataset):
    summary = summary
    project_id = "turtles-of-smsrc"
    metadata_fields = (
        "species_guess",
        "observed_on",
        "location",
    )

    def load_segmentation(self, df):
        cols = ['bbox_x', 'bbox_y', 'bbox_w', 'bbox_h']
        segmentation = pd.read_csv(f'{self.root}/segmentation.csv')
        df = pd.merge(df, segmentation, on='image_id', how='outer')
        df['bbox'] = list(df[cols].to_numpy())
        df = df.drop(cols, axis=1)
        df = df.reset_index(drop=True)
        for _, df_id in df.groupby('image_id'):
            image_ids = [f'{image_id}_{i}' for i, image_id in enumerate(df_id['image_id'])]
            df.loc[df_id.index, 'image_id'] = image_ids
        return df

    def fix_labels(self, df):
        replace = [
            # Heads - SP
            (328448963, 322750899),
            (328448962, 321559873),
            (328448961, 326931783),
            (328448960, 328448954),
            (328448954, 326931792),
            (328448953, 328448936),
            (328448947, 326931771),
            (328448946, 322750909),
            (328448933, 325746516),
            (328448932, 326931797),
            (328448928, 328448927),
            (327367333, 327367322),
            (326931797, 326931767),
            (326931796, 326931769),
            (326931791, 321595642),
            (326931785, 326931783),
            (326931783, 321595649),
            (326931778, 325746507),
            (326931769, 325746516),
            (325746515, 321595644),
            (325746512, 322750899),
            (324994639, 321595643), 
            (321595649, 321559873),
            # Heads - ALIKED
            (326931795, 326931765),
            (326931787, 325746520),
            (325746518, 322750920),
            (322750920, 321595643),
            (324994658, 324994646),
            # Heads - DISK
            (326931765, 324994647),
            # Front flippers
            (328448952, 324994647),
            (328448950, 328448948),
            (328448939, 324994656),
            (328448936, 324994660),
            (328448926, 326931775),
            (325746540, 325746506),
            (321595644, 321595639),
            # Front flippers - ALIKED
            (326931764, 325746531),
            (325746532, 325746531),
            (324994660, 324994644),
            # Rear flippers
            (328448927, 325746501),
            (326931779, 322750916),
            (326931774, 326931772),
            (325746516, 324994639),
            (325746510, 322750914),
            (322750924, 322750916),
            # Rear flippers - ALIKED
            (328448965, 328448920),
            (326931781, 325746536),
            # Carapaces
            (328448958, 326931780),
            (328448957, 322750922),
            (324994650, 322750922),
            (322750909, 321559873),
            # Carapaces - ALIKED
            (326931760, 325746505),
            # Carapaces - DISK
            (328448941, 322750925),
        ]
        # 322750909, 325746532, 326931764, 328448960

        replace = sorted(replace, key=lambda row: (row[0], row[1]), reverse=True)
        # TODO: check that the first column is unique
        return self.fix_labels_replace_identity(df, replace)
    
    def create_catalogue(self, load_segmentation=False) -> pd.DataFrame:
        df = pd.read_csv(os.path.join(self.root, "metadata.csv"))
        df["image_id"] = df["observation_id"].astype(str) + "_" + df["photo_id"].astype(str)
        df["identity"] = df["observation_id"]
        df["latitute"] = df["location"].apply(lambda x: ast.literal_eval(x)[0]).astype(float)
        df["longitude"] = df["location"].apply(lambda x: ast.literal_eval(x)[1]).astype(float)
        df = df.drop(["location", "photo_id", "photo_url"], axis=1)
        df = df.rename({
            "observation_id": "encounter_id",
            "species_guess": "species",
            "file_name": "path",
            "observed_on": "date"
            }, axis=1)
        if load_segmentation:
            df = self.load_segmentation(df)

        return self.finalize_catalogue(df)
