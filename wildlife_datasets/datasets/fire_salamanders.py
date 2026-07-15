import os

import numpy as np
import pandas as pd
import requests

from .datasets import WildlifeDataset, utils

# TODO: not finished
summary = {
    "licenses": "Community Data License Agreement – Permissive",
    "licenses_url": "https://cdla.dev/permissive-1-0/",
    "url": "https://figshare.com/articles/figure/Performance_of_different_automatic_photographic_identification_software_for_larvae_and_adults_of_the_European_fire_salamander_-_European_fire_salamander_adult_data_set_Kathleen_Prei_ler/24998690/1",
    "publication_url": "https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0298285",
    "cite": "parham2017animal",
    "animals": {"fire salamader"},
    "animals_simple": "salamanders",
    "real_animals": True,
    "year": 2024,
    "reported_n_total": 377,
    "reported_n_individuals": 311,
    "wild": True,
    "clear_photos": True,
    "pose": "double",
    "unique_pattern": True,
    "from_video": False,
    "cropped": False,
    "span": "12 days",
    "size": 1300,
}


class FireSalamanders(WildlifeDataset):
    summary = summary

    @classmethod
    def _download(cls):
        root_images = "images"
        os.makedirs(root_images, exist_ok=True)
        article = requests.get("https://api.figshare.com/v2/articles/24998690").json()
        files = article.get("files", [])
        for file in files:
            file_save = os.path.join(root_images, file["name"])
            utils.download_url(file["download_url"], file_save)

    @classmethod
    def _extract(cls):
        pass

    def create_catalogue(self) -> pd.DataFrame:
        # Find all images in root
        assert self.root is not None
        data = utils.find_images(self.root)
        data["path"] = data["path"] + os.path.sep + data["file"]
        data["identity"] = data["file"].apply(lambda x: os.path.splitext(x)[0])
        data["image_id"] = utils.get_persistent_id(data["path"])

        # Load the identities to be merged
        identity_table = pd.read_csv(os.path.join(self.root, "identity.csv"), header=None)

        # Make a simple check that the identities are the same
        identities1 = data["identity"].unique()
        identities2 = identity_table.iloc[:, 0].unique()
        if not np.array_equal(np.sort(identities1), np.sort(identities2)):
            raise ValueError("Identities are wrong")

        # Convert the identities into a required format
        identity_groups = []
        for _, row in identity_table.iterrows():
            identities = row.to_list()
            identities = [x.strip() for x in identities if not pd.isnull(x)]
            identities = [x for x in identities if x != ""]
            if len(identities) > 1:
                identity_groups.append(tuple(identities))

        # Finalize the dataframe
        data = self.fix_labels_group_identity(data, identity_groups)
        data = data.drop("file", axis=1)
        return self.finalize_catalogue(data)