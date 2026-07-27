import os

import pandas as pd
from PIL import Image

from . import utils
from .datasets import WildlifeDataset
from .downloads import DownloadKaggle

summary = {
    "licenses": "Attribution 4.0 International (CC BY 4.0)",
    "licenses_url": "https://creativecommons.org/licenses/by/4.0/",
    "url": "https://www.kaggle.com/datasets/timost1234/cat-individuals",
    "publication_url": None,
    "cite": "catindividuals",
    "animals": {"cat"},
    "animals_simple": "cats",
    "real_animals": True,
    "year": 2020,
    "reported_n_total": 13536,
    "reported_n_individuals": 518,
    "wild": False,
    "clear_photos": True,
    "pose": "multiple",
    "unique_pattern": True,
    "from_video": False,
    "cropped": True,
    "span": "short",
    "size": 11000,
}


class CatIndividualImages(DownloadKaggle, WildlifeDataset):
    summary = summary
    kaggle_url = "timost1234/cat-individuals"
    kaggle_type = "datasets"

    @classmethod
    def _extract(cls):
        super()._extract()
        cls._convert_heic_to_jpg()

    @classmethod
    def _convert_heic_to_jpg(cls):
        try:
            import pillow_heif
        except ImportError as e:
            raise ImportError(
                "Loading CatIndividualImages requires pillow-heif to convert its HEIC images. "
                "Install it via: pip install wildlife_datasets[full]"
            ) from e
        pillow_heif.register_heif_opener()

        for path, _, files in os.walk("."):
            for file in files:
                if file.lower().endswith(".heic"):
                    heic_path = os.path.join(path, file)
                    jpg_path = os.path.splitext(heic_path)[0] + ".jpg"
                    if os.path.exists(jpg_path):
                        continue
                    with Image.open(heic_path) as img:
                        img.convert("RGB").save(jpg_path, "JPEG", quality=100)

    def create_catalogue(self) -> pd.DataFrame:
        # Find all images in root
        root = self.get_root()
        data = utils.find_images(root)
        folders = data["path"].str.split(os.path.sep, expand=True)

        # Remove 85 duplicate images
        idx = folders[2].isnull()
        data = data[idx]
        folders = folders[idx]

        # Finalize the dataframe
        df = pd.DataFrame(
            {
                "image_id": data["file"].apply(lambda x: os.path.splitext(x)[0]),
                "path": data["path"] + os.path.sep + data["file"],
                "identity": folders[1].astype(int),
            }
        )
        return self.finalize_catalogue(df)
