import os

import pandas as pd

from .datasets import WildlifeDataset
from .downloads import DownloadURL
from .utils import find_images, parse_bbox_mask

summary = {
    "licenses": "Attribution 4.0 International (CC BY 4.0)",
    "licenses_url": "https://creativecommons.org/licenses/by/4.0/",
    "url": "https://zenodo.org/records/18000809",
    "publication_url": "https://bmva-archive.org.uk/bmvc/2025/assets/workshops/MVCC/Paper_13/paper.pdf",
    "cite": "rosa2025gcn",
    "animals": {"great crested newt"},
    "animals_simple": "newts",
    "real_animals": True,
    "year": 2025,
    "reported_n_total": 1232,
    "reported_n_individuals": 206,
    "wild": False,
    "clear_photos": True,
    "pose": "single",
    "unique_pattern": True,
    "from_video": False,
    "cropped": False,
    "span": "1 month",
    "size": 805,
}


def convert_bbox(s: str | float) -> list[float] | None:
    if pd.isnull(s):
        return None
    x1, y1, x2, y2 = parse_bbox_mask(s)
    return [x1, y1, x2 - x1, y2 - y1]


def convert_rle(s: str | float) -> dict | None:
    if pd.isnull(s):
        return None
    size_str, counts = s.split(":", 1)
    height, width = map(int, size_str.split("x"))
    return {"size": [height, width], "counts": counts.strip("'")}


class GCN_ID(DownloadURL, WildlifeDataset):
    summary = summary
    downloads = [
        ("https://zenodo.org/records/18000809/files/Raw_Data.zip?download=1", "Raw_Data.zip"),
        ("https://zenodo.org/records/18000809/files/metadata.csv?download=1", "metadata.csv"),
    ]

    def create_catalogue(self) -> pd.DataFrame:
        """
        Create the catalogue DataFrame for the GCN_ID dataset.

        The identity is taken from the `identity` column, which follows the paper
        (no recaptures). The metadata also contains `recapture_id`, which merges some
        identities across surveys. It is kept as an extra column, but it is unreliable
        (visual check shows some merged identities having different patterns).

        Returns:
            pd.DataFrame: A dataframe containing one row per image.
            The dataframe includes columns:

                - image_id (int): Unique image identifier.
                - identity (str): Individual identity label.
                - path (str): Relative path to the image file.
                - bbox (list): Bounding box [x, y, w, h].
                - segmentation (dict): Segmentation mask in the RLE format.
                - recapture_id (int): Unverified linking of identities across surveys.
                - survey (int): Survey (encounter) number.
        """

        root = self.get_root()
        data = find_images(root)
        folders = data["path"].str.split(os.path.sep, expand=True)
        df_images = pd.DataFrame(
            {
                "path": data["path"] + os.path.sep + data["file"],
                "code": folders.iloc[:, -1].astype(str) + data["file"].astype(str),
            }
        )

        df = pd.read_csv(os.path.join(root, "metadata.csv"))
        df["code"] = df["identity"].astype(str) + df["file_name"]
        df = pd.merge(df, df_images, on="code")
        df = df.rename(
            {
                "Unnamed: 0": "image_id",
                "segmentation_mask_rle": "segmentation",
            },
            axis=1,
        )
        df = df.drop(["file_name", "code"], axis=1)
        df["bbox"] = df["bbox"].apply(convert_bbox)
        df["segmentation"] = df["segmentation"].apply(convert_rle)

        return self.finalize_catalogue(df)
