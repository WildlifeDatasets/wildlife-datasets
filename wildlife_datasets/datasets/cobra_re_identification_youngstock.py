import os

import pandas as pd

from . import utils
from .datasets import WildlifeDataset
from .downloads import DownloadURL

summary = {
    "licenses": "Attribution 4.0 International (CC BY 4.0)",
    "licenses_url": "https://creativecommons.org/licenses/by/4.0/",
    "url": "https://zenodo.org/records/15018518",
    "publication_url": "https://www.mdpi.com/1424-8220/25/10/2971",
    "cite": "perneel2025dynamic",
    "animals": {"cow"},
    "animals_simple": "cows",
    "real_animals": True,
    "year": 2025,
    "reported_n_total": 11438,
    "reported_n_individuals": 48,
    "wild": False,
    "clear_photos": True,
    "pose": "multiple",
    "unique_pattern": True,
    "from_video": False,
    "cropped": True,
    "span": "2 years",
    "size": 732,
    "description": "The data was collected at a commercial Belgian dairy farm between December 2021 and October 2023 in pen housing eight heifers aged 8 to 12 months. During the experimental period, six groups of animals were monitored, resulting in a total of 48 animals. Video footage was collected continuously during day and night with a roof-mounted 4MP camera (Hikvision Europe, Hoofddorp, The Netherlands, model DS-2CD2643G2-IZS) at 2 fps. For this study, only daylight video recordings were retained. Of the 48 monitored animals, 29 were purebred red Holstein-Friesian and 19 were purebred Holstein-Friesian.",
}


class CoBRAReIdentificationYoungstock(DownloadURL, WildlifeDataset):
    summary = summary
    url = "https://zenodo.org/records/15018518/files/re_identification_youngstock.zip?download=1"
    archive = "re_identification_youngstock.zip"

    def create_catalogue(self) -> pd.DataFrame:
        # Find all images in root
        root = self.get_root()
        data = utils.find_images(root)
        folders = data["path"].str.split(os.path.sep, expand=True)

        # Extract information
        identity = folders.iloc[:, -1]
        orientation = folders.iloc[:, -2]

        # Finalize the dataframe
        df = pd.DataFrame(
            {
                "image_id": utils.create_id(identity + orientation + data["file"]),
                "path": data["path"] + os.path.sep + data["file"],
                "identity": identity,
                "orientation": orientation,
            }
        )
        return self.finalize_catalogue(df)
