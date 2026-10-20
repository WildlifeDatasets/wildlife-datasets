import ast

import pandas as pd

from .downloads import DownloadINaturalist
from .general import Dataset_Metadata

summary = {
    "licenses": "",
    "licenses_url": "",
    "url": "https://www.inaturalist.org/projects/turtles-of-smsrc",
    "publication_url": None,
    "cite": "",
    "animals": {"Green Sea Turtle"},
    "animals_simple": "sea turtles",
    "real_animals": True,
    "original_dataset": True,
    "year": 2025,
    "reported_n_total": None,
    "reported_n_individuals": None,
    "wild": True,
    "clear_photos": True,
    "pose": "multiple",
    "unique_pattern": True,
    "from_video": False,
    "cropped": False,
    "span": "1 year",
    "size": None,
    "description": "The Sharjah Marine Science Research Centre (SMSRC) has created a participatory research project aimed at documenting the occurrence, distribution, and habitat use of marine turtles along the Khor Fakkan coast. Through systematic photographic records, the project seeks to create a visual and georeferenced database of individual turtles observed by researchers, students, divers, and citizen scientists. Each turtle photograph will be tagged with location (GPS), date, and habitat type, allowing for the analysis of spatial patterns, site fidelity, and potential threats such as entanglement or pollution. Over time, the dataset will serve as a baseline for long-term monitoring and for developing awareness materials and conservation strategies for Sharjah’s marine biodiversity.",
}


class TurtlesOfSMSRC(DownloadINaturalist, Dataset_Metadata):
    summary = summary
    project_id = "turtles-of-smsrc"
    metadata_fields = (
        "species_guess",
        "observed_on",
        "location",
    )

    def create_catalogue(self, *, replace_identity: list[tuple] | None = None, **kwargs) -> pd.DataFrame:
        self.replace_identity = [] if replace_identity is None else replace_identity
        return super().create_catalogue(**kwargs)

    def modify_metadata(self, metadata: pd.DataFrame) -> pd.DataFrame:
        metadata["image_id"] = metadata["observation_id"].astype(str) + "_" + metadata["photo_id"].astype(str)
        metadata["identity"] = metadata["observation_id"]
        mask = ~metadata["location"].isnull()
        metadata["latitute"] = metadata.loc[mask, "location"].apply(lambda x: ast.literal_eval(x)[0]).astype(float)
        metadata["longitude"] = metadata.loc[mask, "location"].apply(lambda x: ast.literal_eval(x)[1]).astype(float)
        metadata = metadata.drop(["location", "photo_id", "photo_url"], axis=1)
        metadata = metadata.rename(
            {"observation_id": "encounter_id", "species_guess": "species", "file_name": "path", "observed_on": "date"},
            axis=1,
        )
        return metadata

    def fix_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fix_labels_group_identity(df, self.replace_identity)
