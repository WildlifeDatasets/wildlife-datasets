import pandas as pd

from .downloads import DownloadURL
from .general import Dataset_Metadata
from .utils import get_persistent_id

summary = {
    "licenses": "Attribution 4.0 International (CC BY 4.0)",
    "licenses_url": "https://creativecommons.org/licenses/by/4.0/",
    "url": "https://zenodo.org/records/21605650",
    "publication_url": "https://dl.gi.de/handle/20.500.12116/45701",
    "cite": "moller2025automatisierte",
    "animals": {"cow"},
    "animals_simple": "cows",
    "real_animals": True,
    "original_dataset": True,
    "year": 2026,
    "reported_n_total": 80260,
    "reported_n_individuals": 110,
    "wild": False,
    "clear_photos": True,
    "pose": "single",
    "unique_pattern": True,
    "from_video": True,
    "cropped": False,
    "span": "1.7 years",
    "size": 25100,
    "description": "SideViewCows2026 contains right-side images of dairy cows, predominantly Holstein Friesian with a few Fleckvieh, Jersey and Brown Swiss animals. It consists of three nested subsets with consistent identities: parlor (54,393 video frames of 110 cows from a fixed camera at the milking parlor entrance, cropped to a region of interest), barn (25,260 frames of 69 cows from handheld videos in the barn) and snapshots (607 photographs of 63 cows in varied indoor and outdoor settings). Every image comes with a binary segmentation mask of the animal. Regions with people were overwritten in magenta. Since frames from the same recording are strongly correlated, splits should be grouped by video.",
}


class SideViewCows2026(DownloadURL, Dataset_Metadata):
    summary = summary
    downloads = [
        ("https://zenodo.org/records/21605650/files/parlor.zip?download=1", "parlor.zip"),
        ("https://zenodo.org/records/21605650/files/barn.zip?download=1", "barn.zip"),
        ("https://zenodo.org/records/21605650/files/snapshots.zip?download=1", "snapshots.zip"),
        ("https://zenodo.org/records/21605650/files/manifest.csv?download=1", "manifest.csv"),
    ]
    extract_add_folder = False

    def create_catalogue(self, file_name: str = "manifest.csv") -> pd.DataFrame:
        """
        Create the catalogue DataFrame for the SideViewCows2026 dataset.

        The recordings are identified by the capture time, which is shared by all
        frames of one video. Snapshots are individual photographs and have no video.

        Returns:
            pd.DataFrame: A dataframe containing one row per image.
            The dataframe includes columns:

                - image_id (str): Unique image identifier.
                - identity (int): Individual identity label.
                - path (str): Relative path to the image file.
                - segmentation (str): Relative path to the binary mask.
                - split (str): Gallery (parlor) or query (snapshots), missing for barn.
                - video (int): Index of the video recording, missing for snapshots.
                - frame_number (int): Consecutive number of the image within the individual and subset.
                - time_offset_s (int): Capture time in seconds relative to the earliest image.
        """

        return super().create_catalogue(file_name=file_name)

    def modify_metadata(self, metadata: pd.DataFrame) -> pd.DataFrame:
        metadata = metadata.rename(
            {
                "image_path": "path",
                "mask_path": "segmentation",
                "individual_id": "identity",
                "frame_no": "frame_number",
            },
            axis=1,
        )
        metadata["image_id"] = get_persistent_id(metadata["path"])
        is_video = metadata["subset"] != "snapshots"
        video = metadata.loc[is_video, ["subset", "time_offset_s"]].astype(str).agg("_".join, axis=1)
        metadata["video"] = pd.Series(pd.factorize(video)[0], index=video.index).astype("Int64")
        metadata["split"] = metadata.pop("subset").map({"parlor": "gallery", "snapshots": "query"})
        metadata = metadata.drop(["width", "height", "sha256"], axis=1)
        return metadata
