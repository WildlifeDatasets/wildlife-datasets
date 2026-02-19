import json
import os

import numpy as np
import pandas as pd

from .datasets import WildlifeDataset
from .downloads import DownloadKaggle
from .utils import parse_bbox_mask

summary_2022 = {
    "licenses": "Other",
    "licenses_url": "https://www.kaggle.com/datasets/wildlifedatasets/seaturtleid2022",
    "url": "https://www.kaggle.com/datasets/wildlifedatasets/seaturtleid2022",
    "publication_url": "https://openaccess.thecvf.com/content/WACV2024/html/Adam_SeaTurtleID2022_A_Long-Span_Dataset_for_Reliable_Sea_Turtle_Re-Identification_WACV_2024_paper.html",
    "cite": "adam2024seaturtleid2022",
    "animals": {"loggerhead turtle"},
    "animals_simple": "sea turtles",
    "real_animals": True,
    "year": 2022,
    "reported_n_total": 8729,
    "reported_n_individuals": 438,
    "wild": True,
    "clear_photos": False,
    "pose": "multiple",
    "unique_pattern": True,
    "from_video": False,
    "cropped": False,
    "span": "12 years",
    "size": 2000,
}

summary_heads = {
    "licenses": "Other",
    "licenses_url": "https://www.kaggle.com/datasets/wildlifedatasets/seaturtleidheads",
    "url": "https://www.kaggle.com/datasets/wildlifedatasets/seaturtleidheads",
    "publication_url": "https://openaccess.thecvf.com/content/WACV2024/html/Adam_SeaTurtleID2022_A_Long-Span_Dataset_for_Reliable_Sea_Turtle_Re-Identification_WACV_2024_paper.html",
    "cite": "adam2024seaturtleid2022",
    "animals": {"loggerhead turtle"},
    "animals_simple": "sea turtles",
    "real_animals": True,
    "year": 2022,
    "reported_n_total": 7582,
    "reported_n_individuals": 400,
    "wild": True,
    "clear_photos": False,
    "pose": "multiple",
    "unique_pattern": True,
    "from_video": False,
    "cropped": True,
    "span": "9 years",
    "size": 425,
}


class SeaTurtleID2022(DownloadKaggle, WildlifeDataset):
    summary = summary_2022
    kaggle_url = "wildlifedatasets/seaturtleid2022"
    kaggle_type = "datasets"

    def create_catalogue(self, category_name="head") -> pd.DataFrame:
        """Creates dataframe for SeaTurtleID2022.

        Args:
            category_name (str, optional): Choose one from ['turtle', 'flipper', 'head'].

        Returns:
            Created dataframe.
        """

        assert self.root is not None
        # Load annotations JSON file
        path_json = os.path.join("turtles-data", "data", "annotations.json")
        with open(os.path.join(self.root, path_json)) as file:
            data = json.load(file)
        path_csv = os.path.join("turtles-data", "data", "metadata_splits.csv")
        with open(os.path.join(self.root, path_csv)) as file:
            df_images = pd.read_csv(file)
        # Extract categories
        categories = {}
        for category in data["categories"]:
            categories[category["name"]] = category["id"]
        if category_name == "flipper":
            orientation_col = "location"
        else:
            orientation_col = "orientation"
        if category_name not in categories:
            # printfor category in categories.keys()
            raise Exception(f"Category {category_name} not allowed. Choose one from {categories.keys()}.")
        category_id = categories[category_name]

        # Extract data from the JSON file
        create_dict = lambda i: {
            "id": i["id"],
            "bbox": i["bbox"],
            "image_id": i["image_id"],
            "segmentation": i["segmentation"],
            "orientation": i["attributes"][orientation_col] if orientation_col in i["attributes"] else np.nan,
        }
        df_annotation = pd.DataFrame([create_dict(i) for i in data["annotations"] if i["category_id"] == category_id])
        idx_bbox = ~df_annotation["bbox"].isnull()
        df_annotation.loc[idx_bbox, "bbox"] = df_annotation.loc[idx_bbox, "bbox"].apply(parse_bbox_mask)
        df_images.rename({"id": "image_id"}, axis=1, inplace=True)

        # Merge the information from the JSON file
        df = pd.merge(df_images, df_annotation, on="image_id", how="outer")
        df["path"] = "turtles-data" + os.path.sep + "data" + os.path.sep + df["file_name"].str.replace("/", os.path.sep)
        df = df.drop(
            ["id", "file_name", "timestamp", "width", "height", "year", "split_closed_random", "split_open"], axis=1
        )
        df.rename({"split_closed": "original_split"}, axis=1, inplace=True)
        df["date"] = df["date"].apply(lambda x: x[:4] + "-" + x[5:7] + "-" + x[8:10])

        df["image_id"] = range(1, len(df) + 1)
        return self.finalize_catalogue(df)


class SeaTurtleIDHeads(DownloadKaggle, WildlifeDataset):
    summary = summary_heads
    kaggle_url = "wildlifedatasets/seaturtleidheads"
    kaggle_type = "datasets"

    def create_catalogue(self) -> pd.DataFrame:
        assert self.root is not None
        # Load annotations JSON file
        path_json = "annotations.json"
        with open(os.path.join(self.root, path_json)) as file:
            data = json.load(file)

        # Extract dtaa from the JSON file
        create_dict = lambda i: {
            "id": i["id"],
            "image_id": i["image_id"],
            "identity": i["identity"],
            "orientation": i["position"],
        }
        df_annotation = pd.DataFrame([create_dict(i) for i in data["annotations"]])
        create_dict = lambda i: {"file_name": i["path"].split("/")[-1], "image_id": i["id"], "date": i["date"]}
        df_images = pd.DataFrame([create_dict(i) for i in data["images"]])

        # Merge the information from the JSON file
        df = pd.merge(df_annotation, df_images, on="image_id")
        df["path"] = "images" + os.path.sep + df["identity"] + os.path.sep + df["file_name"]
        df = df.drop(["image_id", "file_name"], axis=1)
        df["date"] = df["date"].apply(lambda x: x[:4] + "-" + x[5:7] + "-" + x[8:10])

        # Finalize the dataframe
        df.rename({"id": "image_id"}, axis=1, inplace=True)
        return self.finalize_catalogue(df)
