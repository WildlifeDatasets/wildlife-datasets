import os


import numpy as np
import pandas as pd


from .datasets import utils, WildlifeDataset


def restrict(data, folders, idx):
    data, folders = data[idx], folders[idx]
    while True:
        max_col = np.max(folders.columns)
        if all(folders[max_col].isnull()):
            folders = folders.drop(max_col, axis=1)
        else:
            break
    return data, folders


def get_name(x):
    x = x.upper().replace('-', '_')
    x_splits = x.split('_')
    if len(x_splits) >= 2 and ('IMG_' not in x or len(x_splits) >= 3):
        y = x_splits[-1]
        for a in ['.', ' ', '(', ')']:
            y = y.split(a)[0]
        return y.strip()
    return None


class NewtsKent(WildlifeDataset):
    def create_catalogue(self, load_segmentation=False):
        data = utils.find_images(self.root)
        folders = data["path"].str.split(os.path.sep, expand=True)
        # if (
        #     folders[1].nunique() != 1
        #     and folders[1].iloc[0] != "Identification"
        # ):
        #     raise ValueError("Structure wrong")
        # idx = folders[3].isnull()
        # data, folders = restrict(data, folders, idx)

        data["identity"] = data["file"].apply(get_name)

        idx = ~folders[2].isnull()
        data, folders = restrict(data, folders, idx)
        idx = ~folders[2].apply(lambda x: x.startswith("Duplicated"))
        data, folders = restrict(data, folders, idx)
        # TODO: no idea what to do with these

        # TODO: possibly removing too many. now better than keeping bad
        idx = ~data["identity"].isnull()
        data, folders = restrict(data, folders, idx)
        # TODO: removing juveniles and other. will not work when 10k+ individuals are there
        idx = data["identity"].apply(len) <= 5
        data, folders = restrict(data, folders, idx)

        data["path"] = data["path"] + os.path.sep + data["file"]
        data["image_id"] = utils.create_id(
            data["path"].apply(lambda x: x.replace(os.path.sep, "/"))
        ).astype(str)
        # data["year"] = folders[0].apply(lambda x: int(x[:4]))
        data = data.drop("file", axis=1)

        if load_segmentation:
            cols = ["bbox_x", "bbox_y", "bbox_w", "bbox_h"]
            segmentation = pd.read_csv(f"{self.root}/segmentation.csv")
            data = pd.merge(data, segmentation, on="image_id", how="outer")
            data["bbox"] = list(data[cols].to_numpy())
            data["segmentation"] = data["segmentation"].apply(
                lambda x: eval(x)
            )
            data = data.drop(cols, axis=1)
            data = data.reset_index(drop=True)

        return self.finalize_catalogue(data)