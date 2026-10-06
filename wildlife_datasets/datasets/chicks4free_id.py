import numpy as np
import pandas as pd
from datasets import load_dataset

from .datasets import WildlifeDataset
from .downloads import DownloadHuggingFace

summary = {
    "licenses": "Attribution 4.0 International (CC BY 4.0)",
    "licenses_url": "https://creativecommons.org/licenses/by/4.0/",
    "url": "https://huggingface.co/datasets/dariakern/Chicks4FreeID",
    "publication_url": None,
    "cite": "kern2024towards",
    "animals": {"chickens"},
    "animals_simple": "chickens",
    "real_animals": True,
    "year": 2024,
    "reported_n_total": 1146,
    "reported_n_individuals": 50,
    "wild": False,
    "clear_photos": True,
    "pose": "single",
    "unique_pattern": False,
    "from_video": False,
    "cropped": True,
    "span": "short",
    "size": 1401,
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
