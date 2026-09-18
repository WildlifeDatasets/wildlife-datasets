import numpy as np
import pandas as pd
from datasets import load_dataset

from .datasets import WildlifeDataset
from .downloads import DownloadHuggingFace

# Shared base summary for gorilla datasets
_gorilla_summary_base = {
    "animals": {"gorilla"},
    "animals_simple": "gorillas",
    "real_animals": True,
    "clear_photos": True,
    "unique_pattern": False,
    "pose": "single",
    "from_video": True,
    "cropped": True,
    "licenses": "Attribution 4.0 International (CC BY 4.0)",
    "licenses_url": "https://creativecommons.org/licenses/by/4.0/",
    "publication_url": "https://arxiv.org/abs/2512.07776",
    "cite": "schall2026gorillawatch",
    "year": 2026,
}

summary_wild = {
    **_gorilla_summary_base,
    "url": "https://huggingface.co/datasets/gorilla-watch/Gorilla-SPAC-Wild",
    "reported_n_total": 160818,
    "reported_n_individuals": 135,
    "wild": True,
    "span": "4.5 years",
    "size": 59440,
}

summary_zoo = {
    **_gorilla_summary_base,
    "url": "https://huggingface.co/datasets/gorilla-watch/Gorilla-Zoo-Berlin",
    "reported_n_total": 188679,
    "reported_n_individuals": 5,
    "wild": False,
    "span": "2 months",
    "size": 12530,
}


class GorillaWatchWild(DownloadHuggingFace, WildlifeDataset):
    summary = summary_wild
    hf_url = "gorilla-watch/Gorilla-SPAC-Wild"

    @classmethod
    def _download(cls, config="face_with_body"):
        super()._download(config)

    def create_catalogue(self, config="face_with_body") -> pd.DataFrame:
        dataset = load_dataset(self.hf_url, config)

        dfs = []
        for split in dataset.keys():
            n_rows = dataset[split].num_rows
            df = pd.DataFrame(
                {
                    "identity": dataset[split]["class"],
                    "path": np.nan,
                    "split_original": [split] * n_rows,
                    "camera": dataset[split]["camera"],
                    "date": dataset[split]["date"],
                    "video_name": dataset[split]["video"],
                    "frame_number": dataset[split]["frame_number"],
                }
            )
            dfs.append(df)

        df = pd.concat(dfs, ignore_index=True)
        df["image_id"] = range(len(df))
        df["video"] = pd.factorize(df["video_name"])[0]

        self.dataset = dataset
        self.config = config
        return self.finalize_catalogue(df)

    def get_image(self, idx):
        # Map flat index to split and local index
        split_names = list(self.dataset.keys())
        split_sizes = [self.dataset[split].num_rows for split in split_names]
        cumsum = np.cumsum(split_sizes)

        idx = self._normalize_idx(idx)
        split_idx = np.searchsorted(cumsum, idx, side="right")
        local_idx = idx if split_idx == 0 else idx - cumsum[split_idx - 1]

        split = split_names[split_idx]
        return self.dataset[split][int(local_idx)]["image"]


class GorillaZooBerlin(DownloadHuggingFace, WildlifeDataset):
    summary = summary_zoo
    hf_url = "gorilla-watch/Gorilla-Zoo-Berlin"

    @classmethod
    def _download(cls, config="face_with_body"):
        super()._download(config)

    def create_catalogue(self, config="face_with_body") -> pd.DataFrame:
        dataset = load_dataset(self.hf_url, config)

        n_rows = dataset["test"].num_rows
        df = pd.DataFrame(
            {
                "image_id": range(n_rows),
                "identity": dataset["test"]["class"],
                "path": np.nan,
                "split_original": ["test"] * n_rows,
                "camera": dataset["test"]["camera"],
                "date": dataset["test"]["date"],
                "time": dataset["test"]["time"],
                "video_name": dataset["test"]["video"],
                "frame_number": dataset["test"]["frame_number"],
            }
        )
        df["video"] = pd.factorize(df["video_name"])[0]

        self.dataset = dataset
        self.config = config
        return self.finalize_catalogue(df)

    def get_image(self, idx):
        idx = self._normalize_idx(idx)
        return self.dataset["test"][idx]["image"]
