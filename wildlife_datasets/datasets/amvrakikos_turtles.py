import os

import pandas as pd

from . import utils
from .datasets import WildlifeDataset
from .downloads import DownloadKaggle

summary = {
    "licenses": "Other",
    "licenses_url": "https://www.kaggle.com/datasets/wildlifedatasets/amvrakikosturtles",
    "url": "https://www.kaggle.com/datasets/wildlifedatasets/amvrakikosturtles",
    "publication_url": "https://www.biorxiv.org/content/10.1101/2024.09.13.612839",
    "cite": "adam2024exploiting",
    "animals": {"loggerhead turtle"},
    "animals_simple": "sea turtles",
    "real_animals": True,
    "year": 2024,
    "reported_n_total": 200,
    "reported_n_individuals": 50,
    "wild": True,
    "clear_photos": True,
    "pose": "double",
    "unique_pattern": True,
    "from_video": False,
    "cropped": False,
    "span": "4.4 years",
    "size": 1870,
    "description": "This dataset consists of photographs of Mediterranean loggerhead sea turtles taken at Amvrakikos Gulf, Greece (39°02'N, 21°06'E) which is wellknown foraging site for adult and juvenile turtles. Photographs were collected as part of a long-term capture-mark-recapture project conducted by ARCHELON, the Sea Turtle Protection Society of Greece. Turtles were captured from a boat using the sea turtle rodeo technique and among other data collected, photographs of the head sides were taken while the animal was on the boat. In all photographs, either the whole side of the head was fully shaded or fully illuminated by the sun. All photographs in this dataset were taken during the summer months (June-August) between 2014 and 2022, using a selection of different digital cameras of varying optical resolution.",
}


class AmvrakikosTurtles(DownloadKaggle, WildlifeDataset):
    summary = summary
    kaggle_url = "wildlifedatasets/amvrakikosturtles"
    kaggle_type = "datasets"

    def create_catalogue(self) -> pd.DataFrame:
        root = self.get_root()
        data = pd.read_csv(os.path.join(root, "annotations.csv"))

        # Get the bounding box
        columns_bbox = ["bbox_x", "bbox_y", "bbox_width", "bbox_height"]
        bbox = data[columns_bbox].to_numpy()
        bbox = pd.Series(list(bbox))
        path = "images" + os.path.sep + data["image_name"]
        date = [utils.get_image_date(os.path.join(root, p)) for p in path]

        # Finalize the dataframe
        df = pd.DataFrame(
            {
                "image_id": range(len(data)),
                "path": path,
                "identity": data["image_name"].apply(lambda x: x.split("_")[0]).astype(int),
                "date": date,
                "orientation": data["image_name"].apply(lambda x: x.split("_")[2]),
                "bbox": bbox,
            }
        )
        df = df[df["orientation"] != "top"]
        df["image_id"] = range(len(df))
        return self.finalize_catalogue(df)
