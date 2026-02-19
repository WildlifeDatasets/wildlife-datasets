import os

import numpy as np
import pandas as pd

from .datasets import WildlifeDataset
from .downloads import DownloadKaggle

summary = {
    "licenses": "Other",
    "licenses_url": "https://www.kaggle.com/datasets/wildlifedatasets/reunionturtles",
    "url": "https://www.kaggle.com/datasets/wildlifedatasets/reunionturtles",
    "publication_url": "https://www.biorxiv.org/content/10.1101/2024.09.13.612839",
    "cite": "adam2024exploiting",
    "animals": {"green turtle", "hawksbill turtle"},
    "animals_simple": "sea turtles",
    "real_animals": True,
    "year": 2024,
    "reported_n_total": 336,
    "reported_n_individuals": 84,
    "wild": True,
    "clear_photos": True,
    "pose": "double",
    "unique_pattern": True,
    "from_video": False,
    "cropped": True,
    "span": "4.2 years",
    "size": 32,
}


class ReunionTurtles(DownloadKaggle, WildlifeDataset):
    summary = summary
    kaggle_url = "wildlifedatasets/reunionturtles"
    kaggle_type = "datasets"

    def create_catalogue(self) -> pd.DataFrame:
        assert self.root is not None
        data = pd.read_csv(os.path.join(self.root, "data.csv"))

        date = pd.to_datetime(data["Date"])
        year = date.apply(lambda x: x.year)
        path = (
            data["Species"]
            + os.path.sep
            + data["Turtle_ID"]
            + os.path.sep
            + year.astype(str)
            + os.path.sep
            + data["Photo_name"]
        )
        orientation = data["Photo_name"].apply(lambda x: os.path.splitext(x)[0].split("_")[2])
        orientation = orientation.replace({"L": "left", "R": "right"})

        # Extract and convert ID codes
        id_code = list(data["ID_Code"].apply(lambda x: x.split(";")))
        max0 = 0
        max1 = 0
        for x in id_code:
            for y in x:
                max0 = max(max0, int(y[0]))
                max1 = max(max1, int(y[1]))
        code = np.zeros((len(id_code), max0, max1), dtype=int)
        for i, x in enumerate(id_code):
            for y in x:
                code[i, int(y[0]) - 1, int(y[1]) - 1] = int(y[2])
        code = code.reshape(len(id_code), -1)

        # Finalize the dataframe
        df = pd.DataFrame(
            {
                "image_id": range(len(data)),
                "path": path,
                "identity": data["Turtle_ID"],
                "date": date,
                "orientation": orientation,
                "species": data["Species"],
                "id_code": list(code),
            }
        )
        return self.finalize_catalogue(df)
