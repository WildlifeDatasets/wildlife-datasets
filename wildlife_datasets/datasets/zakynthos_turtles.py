import os

import pandas as pd

from .datasets import WildlifeDataset
from .downloads import DownloadKaggle

summary = {
    "licenses": "Other",
    "licenses_url": "https://www.kaggle.com/datasets/wildlifedatasets/zakynthosturtles",
    "url": "https://www.kaggle.com/datasets/wildlifedatasets/zakynthosturtles",
    "publication_url": "https://www.biorxiv.org/content/10.1101/2024.09.13.612839",
    "cite": "adam2024exploiting",
    "animals": {"loggerhead turtle"},
    "animals_simple": "sea turtles",
    "real_animals": True,
    "year": 2024,
    "reported_n_total": 160,
    "reported_n_individuals": 40,
    "wild": True,
    "clear_photos": True,
    "pose": "double",
    "unique_pattern": True,
    "from_video": False,
    "cropped": False,
    "span": "2.5 years",
    "size": 826,
}


class ZakynthosTurtles(DownloadKaggle, WildlifeDataset):
    summary = summary
    kaggle_url = "wildlifedatasets/zakynthosturtles"
    kaggle_type = "datasets"

    def create_catalogue(self) -> pd.DataFrame:
        assert self.root is not None
        data = pd.read_csv(os.path.join(self.root, "annotations.csv"))
        bbox = pd.read_csv(os.path.join(self.root, "bbox.csv"))
        data = pd.merge(data, bbox, left_on="path", right_on="image_name")

        dates = data["date"].str.split("_")
        dates = dates.apply(lambda x: x[2] + "-" + x[1] + "-" + x[0])
        df = pd.DataFrame(
            {
                "image_id": range(len(data)),
                "path": "images/" + data["path"],
                "identity": data["identity"],
                "date": dates,
                "orientation": data["orientation"],
                "bbox": data[["bbox_x", "bbox_y", "bbox_width", "bbox_height"]].values.tolist(),
            }
        )
        return self.finalize_catalogue(df)
