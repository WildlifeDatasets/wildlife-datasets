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
    "original_dataset": True,
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
    "n_images": 160818,
    "n_images_identified": 160818,
    "n_individuals": 135,
    "wild": True,
    "span": "4.5 years",
    "size": 59440,
    "description": "GorillaWatchWild (Gorilla-SPAC-Wild) contains 135 individually identified western lowland gorillas extracted from camera-trap video footage collected at Odzala-Kokoua National Park, Republic of Congo. Detections are provided as paired face and body crops obtained via a fine-tuned YOLOv8-Nano detector and tracker, ground-truthed by primate researchers with 15+ years of experience with this population. The dataset has an individual-disjoint train/validation/test split for open-set evaluation, plus a single_encounter split of gorillas observed only once.",
}

summary_zoo = {
    **_gorilla_summary_base,
    "url": "https://huggingface.co/datasets/gorilla-watch/Gorilla-Zoo-Berlin",
    "n_images": 188679,
    "n_images_identified": 188679,
    "n_individuals": 5,
    "wild": False,
    "span": "2 months",
    "size": 12530,
    "description": "GorillaZooBerlin (Gorilla-Zoo-Berlin) is a cross-domain evaluation companion to GorillaWatchWild, part of the same GorillaWatch project. It contains camera-trap footage of the 5 western lowland gorillas of a single social group at Zoo Berlin, used to evaluate how well models trained on in-the-wild footage generalize to a controlled zoo environment with artificial structures, glass and different camera angles.",
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
                    "hf_index": range(n_rows),
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
                "hf_index": range(n_rows),
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
