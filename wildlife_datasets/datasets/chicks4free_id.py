import numpy as np
import pandas as pd
from datasets import load_dataset

from .datasets import WildlifeDataset
from .downloads import DownloadHuggingFace

summary = {
    "licenses": "Attribution 4.0 International (CC BY 4.0)",
    "licenses_url": "https://creativecommons.org/licenses/by/4.0/",
    "url": "https://huggingface.co/datasets/dariakern/Chicks4FreeID",
    "publication_url": "https://www.mdpi.com/2076-2615/15/1/1",
    "cite": "kern2024towards",
    "animals": {"chickens"},
    "animals_simple": "chickens",
    "real_animals": True,
    "original_dataset": True,
    "year": 2024,
    "n_images": 1146,
    "n_images_identified": 1146,
    "n_individuals": 50,
    "wild": False,
    "clear_photos": True,
    "pose": "single",
    "unique_pattern": False,
    "from_video": False,
    "cropped": True,
    "span": "short",
    "size": 1401,
    "description": 'The Chicks4FreeID dataset contains top-down view images of individually segmented and annotated chickens (with roosters and ducks also possibly present and labeled as such). 11 different coops with 54 individuals were visited for manual data collection. Each of the 677 images depicts at least one chicken. The identities of the 50 chickens, 2 roosters and 2 ducks were annotated for a total of 1270 animal instances. Annotation additionally contains visibility ratings of "best", "good", and "bad" for each animal instance.',
}


class Chicks4FreeID(DownloadHuggingFace, WildlifeDataset):
    summary = summary
    hf_url = "dariakern/Chicks4FreeID"
    image_column = "crop"

    @classmethod
    def _download(cls, hf_option="chicken-re-id-all-visibility"):
        super()._download(hf_option)

    def create_catalogue(self, hf_option="chicken-re-id-all-visibility") -> pd.DataFrame:
        dataset = load_dataset(self.hf_url, hf_option)

        n_train = dataset["train"].num_rows
        n_test = dataset["test"].num_rows
        self.dataset = dataset
        df = pd.DataFrame(
            {
                "image_id": range(n_train + n_test),
                "identity": list(dataset["train"]["identity"]) + list(dataset["test"]["identity"]),
                "path": np.nan,
                "split_original": n_train * ["train"] + n_test * ["test"],
                "hf_index": list(range(n_train)) + list(range(n_test)),
            }
        )

        return self.finalize_catalogue(df)
