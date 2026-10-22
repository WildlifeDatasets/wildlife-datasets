import os

import pandas as pd

from . import utils
from .datasets import WildlifeDataset
from .downloads import DownloadURL

summary = {
    "licenses": "Non-Commercial Government Licence for public sector information",
    "licenses_url": "https://www.nationalarchives.gov.uk/doc/non-commercial-government-licence/version/2/",
    "url": "https://data.bris.ac.uk/data/dataset/3owflku95bxsx24643cybxu3qh",
    "publication_url": "https://openaccess.thecvf.com/content_ICCV_2017_workshops/w41/html/Andrew_Visual_Localisation_and_ICCV_2017_paper.html",
    "cite": "andrew2017visual",
    "animals": {"Friesian cattle"},
    "animals_simple": "cows",
    "real_animals": True,
    "original_dataset": True,
    "year": 2017,
    "n_images": 46340,
    "n_images_identified": 46340,
    "n_individuals": 23,
    "wild": False,
    "clear_photos": True,
    "pose": "single",
    "unique_pattern": True,
    "from_video": True,
    "cropped": True,
    "span": "1 day",
    "size": 724,
    "description": "Datasets AerialCattle2017, Cows2021, FriesianCattle2015, FriesianCattle2017, MultiCamCows2024 and OpenCows2020 were created by one group. They capture Holstein-Friesian cows from an aerial standpoint. All images were extracted from videos. FriesianCattle2015 and FriesianCattle2017 were obtained filming cows exiting a milking file. AerialCattle2017 was captured by a drone in an outdoor agricultural field environment. OpenCows2020 combines these datasets. Since the distance of the camera ranges from approximately 4m (FriesianCattle2015) to 25m (AerialCattle2017), it is relatively easy to separate these datasets. Moreover, no individual cow seems to be present in both image acquisitions. Cows2021 depicts the cows in a similar way as FriesianCattle2015, when the camera pointed downwards from 4m above the ground over a walkway between milking parlour and holding pens. Some of the datasets are provided with videos and besides cow re-identification they also aim at cow detection and localization.",
}


class AerialCattle2017(DownloadURL, WildlifeDataset):
    summary = summary
    url = "https://data.bris.ac.uk/datasets/tar/3owflku95bxsx24643cybxu3qh.zip"
    archive = "3owflku95bxsx24643cybxu3qh.zip"

    def create_catalogue(self) -> pd.DataFrame:
        # Find all images in root
        root = self.get_root()
        data = utils.find_images(root)
        folders = data["path"].str.split(os.path.sep, expand=True)

        # Finalize the dataframe
        df = pd.DataFrame(
            {
                "image_id": utils.create_id(data["file"]),
                "path": data["path"] + os.path.sep + data["file"],
                "identity": folders[1].astype(int),
                "video": folders[2].astype(int),
            }
        )
        return self.finalize_catalogue(df)
