import json
import os

import pandas as pd

from . import utils
from .datasets import WildlifeDataset

summary = {
    "licenses": "Attribution-NonCommercial-ShareAlike 4.0 International",
    "licenses_url": "https://creativecommons.org/licenses/by-nc-sa/4.0/",
    "url": "https://lila.science/datasets/atrw",
    "publication_url": "https://arxiv.org/abs/1906.05586",
    "cite": "li2019atrw",
    "animals": {"amur tiger"},
    "animals_simple": "tigers",
    "real_animals": True,
    "year": 2019,
    "reported_n_total": 9496,
    "reported_n_individuals": 92,
    "wild": False,
    "clear_photos": False,
    "pose": "double",
    "unique_pattern": True,
    "from_video": True,
    "cropped": False,
    "span": "short",
    "size": 1760,
}


class ATRW(WildlifeDataset):
    summary = summary
    url = "https://github.com/cvwc2019/ATRWEvalScript/archive/refs/heads/main.zip"
    archive = "main.zip"
    downloads = [
        # Wild dataset (Detection)
        (
            "https://lilawildlife.blob.core.windows.net/lila-wildlife/cvwc2019/test/atrw_detection_test.tar.gz",
            "atrw_detection_test.tar.gz",
        ),
        # Re-ID dataset
        (
            "https://lilawildlife.blob.core.windows.net/lila-wildlife/cvwc2019/train/atrw_reid_train.tar.gz",
            "atrw_reid_train.tar.gz",
        ),
        (
            "https://lilawildlife.blob.core.windows.net/lila-wildlife/cvwc2019/train/atrw_anno_reid_train.tar.gz",
            "atrw_anno_reid_train.tar.gz",
        ),
        (
            "https://lilawildlife.blob.core.windows.net/lila-wildlife/cvwc2019/test/atrw_reid_test.tar.gz",
            "atrw_reid_test.tar.gz",
        ),
        (
            "https://lilawildlife.blob.core.windows.net/lila-wildlife/cvwc2019/test/atrw_anno_reid_test.tar.gz",
            "atrw_anno_reid_test.tar.gz",
        ),
    ]

    @classmethod
    def _download(cls):
        for url, archive in cls.downloads:
            utils.download_url(url, archive)
        # Evaluation scripts
        utils.download_url(cls.url, cls.archive)

    @classmethod
    def _extract(cls):
        for url, archive in cls.downloads:
            archive_name = archive.split(".")[0]
            utils.extract_archive(archive, archive_name, delete=True)
        # Evaluation scripts
        utils.extract_archive(cls.archive, "eval_script", delete=True)

    def create_catalogue(self) -> pd.DataFrame:
        # Load information for the reid_train part of the dataset
        assert self.root is not None
        ids = pd.read_csv(
            os.path.join(self.root, "atrw_anno_reid_train", "reid_list_train.csv"),
            names=["identity", "path"],
            header=None,
        )
        ids["id"] = ids["path"].str.split(".", expand=True)[0].astype(int)
        ids["original_split"] = "train"

        # Load keypoints for the reid_train part of the dataset
        with open(os.path.join(self.root, "atrw_anno_reid_train", "reid_keypoints_train.json")) as file:
            keypoints = json.load(file)
        df_keypoints = {
            "path": pd.Series(keypoints.keys()),
            "keypoints": pd.Series(list(pd.DataFrame([keypoints[key] for key in keypoints.keys()]).to_numpy())),
        }
        data = pd.DataFrame(df_keypoints)

        # Merge information for the reid_train part of the dataset
        df_train = pd.merge(ids, data, on="path", how="left")
        df_train["path"] = "atrw_reid_train" + os.path.sep + "train" + os.path.sep + df_train["path"]

        # Load information for the test_plain part of the dataset
        with open(
            os.path.join(self.root, "eval_script", "ATRWEvalScript-main", "annotations", "gt_test_plain.json")
        ) as file:
            identity = json.load(file)
        identity = pd.DataFrame(identity)
        ids = pd.read_csv(
            os.path.join(self.root, "atrw_anno_reid_test", "reid_list_test.csv"), names=["path"], header=None
        )
        ids["id"] = ids["path"].str.split(".", expand=True)[0].astype(int)
        ids["original_split"] = "test"
        ids = pd.merge(ids, identity, left_on="id", right_on="imgid", how="left")
        ids = ids.drop(["query", "frame", "imgid"], axis=1)
        ids.rename(columns={"entityid": "identity"}, inplace=True)

        # Load keypoints for the test part of the dataset
        with open(os.path.join(self.root, "atrw_anno_reid_test", "reid_keypoints_test.json")) as file:
            keypoints = json.load(file)
        df_keypoints = {
            "path": pd.Series(keypoints.keys()),
            "keypoints": pd.Series(list(pd.DataFrame([keypoints[key] for key in keypoints.keys()]).to_numpy())),
        }
        data = pd.DataFrame(df_keypoints)

        # Merge information for the test_plain part of the dataset
        df_test1 = pd.merge(ids, data, on="path", how="left")
        df_test1["path"] = "atrw_reid_test" + os.path.sep + "test" + os.path.sep + df_test1["path"]

        # Load information for the test_wild part of the dataset
        with open(
            os.path.join(self.root, "eval_script", "ATRWEvalScript-main", "annotations", "gt_test_wild.json")
        ) as file:
            identity = json.load(file)
        ids = utils.find_images(os.path.join(self.root, "atrw_detection_test", "test"))
        ids["imgid"] = ids["file"].str.split(".", expand=True)[0].astype("int")
        entries = []
        for key in identity.keys():
            for entry in identity[key]:
                bbox = [
                    entry["bbox"][0],
                    entry["bbox"][1],
                    entry["bbox"][2] - entry["bbox"][0],
                    entry["bbox"][3] - entry["bbox"][1],
                ]
                entries.append({"imgid": int(key), "bbox": bbox, "identity": entry["eid"]})
        entries = pd.DataFrame(entries)

        # Merge information for the test_wild part of the dataset
        df_test2 = pd.merge(ids, entries, on="imgid", how="left")
        df_test2["path"] = "atrw_detection_test" + os.path.sep + "test" + os.path.sep + df_test2["file"]
        df_test2["id"] = df_test2["imgid"].astype(str) + "_" + df_test2["identity"].astype(str)
        df_test2["original_split"] = "test"
        df_test2 = df_test2.drop(["file", "imgid"], axis=1)

        # Finalize the dataframe
        df = pd.concat([df_train, df_test1, df_test2])
        df["id"] = utils.create_id(df["id"].astype(str))
        df.rename({"id": "image_id"}, axis=1, inplace=True)
        return self.finalize_catalogue(df)
