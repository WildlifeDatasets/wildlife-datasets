import json
import os

import numpy as np
import pandas as pd

from . import utils
from .datasets import WildlifeDataset
from .downloads import DownloadURL

summary = {
    "licenses": "Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)",
    "licenses_url": "https://creativecommons.org/licenses/by-nc-sa/4.0/",
    "url": "https://doi.org/10.25405/data.ncl.c.4982342",
    "publication_url": "https://arxiv.org/abs/2005.13359",
    "cite": "trotter2020ndd20",
    "animals": {"Northumberland dolphin"},
    "animals_simple": "dolphins",
    "real_animals": True,
    "year": 2020,
    "reported_n_total": None,
    "reported_n_individuals": 82,
    "wild": True,
    "clear_photos": True,
    "pose": "multiple",
    "unique_pattern": True,
    "from_video": False,
    "cropped": False,
    "span": "7 years",
    "size": 2441,
    "description": "The Northumberland Dolphin Dataset 2020 (NDD20), is a challenging image dataset as in contains both above and under water photos of two dolphin species taken between 2011 and 2018. The datasets contains both images taken both above and below water. Below water collection efforts consisted of 36 opportunistic surveys of the Farne Deeps. Above water efforts consisted of 27 surveys along a stretch of the Northumberland coast. Above water photographs were taken using a camera from the deck of a small rigid inflatable boat on days of fair weather and good sea conditions. Below water images are screen grabs from high-definition video footage taken with cameras again under good sea conditions. Individuals in the above water images are identified using the structure of the dolphin's dorsal fin. Below water images are less common, but provide additional features for identification such as general colouring, unique body markings, scarring and patterns formed by injury or skin disease. The images contains multiple annotations including dolphin species and approximately 14% of above water images contain segmentation mask for the dolphin fin.",
}


class NDD20(DownloadURL, WildlifeDataset):
    outdated_dataset = True
    summary = summary
    url = "https://data.ncl.ac.uk/ndownloader/files/22774175"
    archive = "NDD20.zip"

    @classmethod
    def _download(cls):
        exception_text = """Dataset must be downloaded manually.
            Check https://wildlifedatasets.github.io/wildlife-datasets/preprocessing#ndd20"""
        raise RuntimeError(exception_text)

    def create_catalogue(self) -> pd.DataFrame:
        # Load information about the above-water dataset
        root = self.get_root()
        with open(os.path.join(root, "ABOVE_LABELS.json")) as file:
            data = json.load(file)

        # Analyze the information about the above-water dataset
        entries = []
        for key in data.keys():
            regions = data[key]["regions"]
            for region in regions:
                if "id" in region["region_attributes"]:
                    identity = region["region_attributes"]["id"]
                else:
                    identity = self.unknown_name
                segmentation = np.zeros(2 * len(region["shape_attributes"]["all_points_x"]))
                segmentation[0::2] = region["shape_attributes"]["all_points_x"]
                segmentation[1::2] = region["shape_attributes"]["all_points_y"]
                entries.append(
                    {
                        "identity": identity,
                        "species": region["region_attributes"]["species"],
                        "out_of_focus": np.nan,
                        "file_name": data[key]["filename"],
                        "reg_type": region["shape_attributes"]["name"],
                        "segmentation": segmentation,
                        "orientation": "above",
                    }
                )

        # Load information about the below-water dataset
        with open(os.path.join(root, "BELOW_LABELS.json")) as file:
            data = json.load(file)

        # Analyze the information about the below-water dataset
        for key in data.keys():
            regions = data[key]["regions"]
            for region in regions:
                if "id" in region["region_attributes"]:
                    identity = region["region_attributes"]["id"]
                else:
                    identity = self.unknown_name
                segmentation = np.zeros(2 * len(region["shape_attributes"]["all_points_x"]))
                segmentation[0::2] = region["shape_attributes"]["all_points_x"]
                segmentation[1::2] = region["shape_attributes"]["all_points_y"]
                entries.append(
                    {
                        "identity": identity,
                        "species": "WBD",
                        "out_of_focus": region["region_attributes"]["out of focus"] == "true",
                        "file_name": data[key]["filename"],
                        "reg_type": region["shape_attributes"]["name"],
                        "segmentation": segmentation,
                        "orientation": "below",
                    }
                )

        # Create the dataframe from entries
        df = pd.DataFrame(entries)
        if len(df.reg_type.unique()) != 1:
            raise ValueError("Multiple segmentation types")

        # Finalize the dataframe
        df["image_id"] = range(len(df))
        df["path"] = df["orientation"].str.upper() + os.path.sep + df["file_name"]
        df = df.drop(["reg_type", "file_name"], axis=1)
        return self.finalize_catalogue(df)


class NDD20v2(NDD20):
    outdated_dataset = False

    def fix_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        root = self.get_root()
        for i, df_row in df.iterrows():
            # Rewrite wrong segmentations. There is no dolphin -> should be deleted.
            # But that would break compability and the identity is unknown anyway.
            if len(df_row["segmentation"]) == 4:
                img = utils.load_image(os.path.join(root, df_row["path"]))
                w, h = img.size
                df.at[i, "segmentation"] = np.array(utils.bbox_segmentation([0, 0, w, h]))
        return df
