import os
import re
import shutil

import pandas as pd
from huggingface_hub import hf_hub_download

from ..splits import TimeProportionSplit
from . import utils
from .datasets import WildlifeDataset

summary_common = {
    "licenses": "Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)",
    "licenses_url": "https://creativecommons.org/licenses/by-nc-sa/4.0/",
    "publication_url": "https://arxiv.org/abs/2607.00804",
    "cite": "kelebek2026spotted",
    "real_animals": True,
    "year": 2026,
    "wild": True,
    "clear_photos": False,
    "pose": "multiple",
    "unique_pattern": True,
    "from_video": False,
    "cropped": False,
    "span": "2 months",
}


class _DownloadHuggingFaceArchive:
    hf_repo: str
    archive: str

    @classmethod
    def _download(cls) -> None:
        archive_path = hf_hub_download(repo_id=cls.hf_repo, filename=cls.archive, repo_type="dataset")
        shutil.copyfile(archive_path, cls.archive)

    @classmethod
    def _extract(cls) -> None:
        utils.extract_archive(cls.archive, delete=True)


class _SpottedDataset(_DownloadHuggingFaceArchive, WildlifeDataset):
    folder_name: str
    species: str
    time_split_ratio: float = 0.8
    unknown_folders: tuple[str, ...] = ()
    orientation_map: dict[str, str] = {}

    @staticmethod
    def _parse_file_metadata(file_name: str) -> dict[str, str]:
        stem = os.path.splitext(file_name)[0]
        metadata = {}

        camera_match = re.search(r"(CAM[^_]+)", stem)
        if camera_match is not None:
            metadata["camera"] = camera_match.group(1)

        date_match = re.search(r"(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})", stem)
        if date_match is not None:
            date, time = date_match.groups()
            metadata["date"] = f"{date} {time.replace('-', ':')}"
            return metadata

        date_match = re.search(r"_(\d{8})_(\d{6})_", stem)
        if date_match is not None:
            date, time = date_match.groups()
            metadata["date"] = f"{date[:4]}-{date[4:6]}-{date[6:]} {time[:2]}:{time[2:4]}:{time[4:]}"

        return metadata

    def _parse_path_metadata(self, path: str) -> dict[str, str]:
        parts = path.split(os.path.sep)
        identity_index = 1 if parts[0] == self.folder_name else 0
        source_identity = parts[identity_index]
        identity = self.unknown_name if source_identity in self.unknown_folders else source_identity

        metadata = {
            "identity": identity,
            "species": self.species,
        }
        next_index = identity_index + 1
        if next_index < len(parts) - 1:
            next_folder = parts[next_index]
            if next_folder in self.orientation_map:
                metadata["orientation"] = self.orientation_map[next_folder]
            elif identity == self.unknown_name:
                metadata["original_identity"] = next_folder
        return metadata

    def create_catalogue(self) -> pd.DataFrame:
        assert self.root is not None
        data = utils.find_images(self.root)
        if data.empty:
            raise FileNotFoundError(f"No images found in {self.root}.")

        records = []
        for _, row in data.iterrows():
            directory = os.path.normpath(row["path"])
            path = row["file"] if directory == "." else os.path.join(directory, row["file"])
            record = {
                "image_id": path,
                "path": path,
                **self._parse_path_metadata(path),
                **self._parse_file_metadata(row["file"]),
            }
            records.append(record)

        df = pd.DataFrame(records).sort_values("path").reset_index(drop=True)
        self._add_time_aware_split(df)
        return self.finalize_catalogue(df)

    def _add_time_aware_split(self, df: pd.DataFrame) -> None:
        splitter = TimeProportionSplit(
            ratio=self.time_split_ratio,
            identity_skip=self.unknown_name,
            col_label=self.col_label,
        )
        idx_train, idx_test = splitter.split(df)[0]
        df["time_split"] = pd.NA
        df.loc[idx_train, "time_split"] = "train"
        df.loc[idx_test, "time_split"] = "test"


class LeopardID102(_SpottedDataset):
    summary = {
        **summary_common,
        "url": "https://huggingface.co/datasets/WildCAT-Datasets/LeopardID102",
        "animals": {"leopard"},
        "animals_simple": "leopards",
        "reported_n_total": 717,
        "reported_n_individuals": 102,
        "size": 865,
    }
    hf_repo = "WildCAT-Datasets/LeopardID102"
    archive = "LeopardID102.zip"
    folder_name = "LeopardID102"
    species = "leopard"
    unknown_folders = ("No_ID",)


class SpottedHyenaID109(_SpottedDataset):
    summary = {
        **summary_common,
        "url": "https://huggingface.co/datasets/WildCAT-Datasets/SpottedHyenaID109",
        "animals": {"spotted hyena"},
        "animals_simple": "hyenas",
        "reported_n_total": 704,
        "reported_n_individuals": 109,
        "size": 704,
    }
    hf_repo = "WildCAT-Datasets/SpottedHyenaID109"
    archive = "SpottedHyenaID109.zip"
    folder_name = "SpottedHyenaID109"
    species = "spotted hyena"
    orientation_map = {"L": "left", "R": "right"}


class SpottedHyenaID415(_SpottedDataset):
    summary = {
        **summary_common,
        "url": "https://huggingface.co/datasets/WildCAT-Datasets/SpottedHyenaID415",
        "animals": {"spotted hyena"},
        "animals_simple": "hyenas",
        "reported_n_total": 1871,
        "reported_n_individuals": 415,
        "size": 2247,
    }
    hf_repo = "WildCAT-Datasets/SpottedHyenaID415"
    archive = "SpottedHyenaID415.zip"
    folder_name = "SpottedHyenaID415"
    species = "spotted hyena"
    unknown_folders = ("unidentifiable",)
