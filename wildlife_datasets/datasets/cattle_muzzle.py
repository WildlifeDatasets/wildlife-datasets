import os

import pandas as pd

from . import utils
from .datasets import WildlifeDataset
from .downloads import DownloadURL

summary = {
    "licenses": None,
    "licenses_url": None,
    "url": "https://cloud.une.edu.au/index.php/s/eMwaHAPK08dCDru",
    "publication_url": "https://www.mdpi.com/2073-4395/11/11/2365",
    "cite": "shojaeipour2021automated",
    "animals": {"cow"},
    "animals_simple": "cows",
    "real_animals": True,
    "original_dataset": True,
    "year": 2021,
    "n_images": 2899,
    "n_individuals": 300,
    "wild": False,
    "clear_photos": True,
    "pose": "single",
    "unique_pattern": True,
    "from_video": True,
    "cropped": True,
    "span": "short",
    "size": 17600,
    "description": "Biometric images consisting of the cattle muzzle and face were collected at the University of New England’s Tullimba Research Feedlot, Kingstown, NSW, Australia. The images were collected during Induction Day in February 2019 when animals were vaccinated and tagged. In total, 300 cattle were inducted, which involved each animal being restrained in a crush restraint with its head placed in a “head scoop” for application of an eye treatment and micro-chipping. Upon its containment in the crush, a photographer stood approximately 1–2 m in front and photographed images of each individual animal in frontal pose from a camera set at 1 m above the ground. The camera was a Canon D800 (Toyko, Japan) equipped with a 18–55 mm lens (Canon EF-S 18–55 mm f/4–5.6 IS STM).",
}


class CattleMuzzle(DownloadURL, WildlifeDataset):
    summary = summary
    url = "https://cloud.une.edu.au/index.php/s/eMwaHAPK08dCDru/download"
    archive = "Cattle Identification (supplementary material).zip"

    def create_catalogue(self) -> pd.DataFrame:
        # Find all images in root
        root = self.get_root()
        data = utils.find_images(root)
        folders = data["path"].str.split(os.path.sep, expand=True)
        n_folders = max(folders.columns)

        # Extract information
        image_id = data["file"].apply(lambda x: x.split("_")[1].split(".")[0])
        identity = folders[n_folders]

        # Finalize the dataframe
        df = pd.DataFrame(
            {
                "image_id": image_id,
                "path": data["path"] + os.path.sep + data["file"],
                "identity": identity,
            }
        )
        return self.finalize_catalogue(df)
