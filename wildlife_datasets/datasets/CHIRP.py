import os
import re

import pandas as pd
import requests

from . import utils
from .datasets import WildlifeDataset


summary = {
    "licenses": "custom license",
    "licenses_url": "https://edmond.mpg.de/file.xhtml?fileId=345213&version=2.0&toolType=PREVIEW",
    "url": "https://doi.org/10.17617/3.GVO4LG",
    "publication_url": "https://openaccess.thecvf.com/content/CVPR2026/papers/Chan_CHIRP_dataset_towards_long-term_individual-level_behavioral_monitoring_of_bird_populations_CVPR_2026_paper.pdf",
    "cite": "chan2026chirp",
    "animals": {"Siberian Jay"},
    "animals_simple": "bird",
    "real_animals": True,
    "year": 2026,
    "reported_n_total": 402750,
    "reported_n_individuals": 183,
    "wild": True,
    "clear_photos": False,
    "pose": "multiple",
    "unique_pattern": False,
    "from_video": True,
    "cropped": True,
    "span": "3 years",
    "size": 19964,
}


class CHIRP(WildlifeDataset):
    summary = summary
    outdated_dataset = False
    base_url = "https://edmond.mpg.de"
    file_id = 345079
    archive = "CHIRP.zip"

    @classmethod
    def _archive_name_from_response(cls, response: requests.Response, archive: str | None) -> str:
        if archive:
            return archive

        disposition = response.headers.get("Content-Disposition", "")
        match = re.search(r'filename="?([^"]+)"?', disposition)
        if match:
            return os.path.basename(match.group(1))
        return cls.archive

    @classmethod
    def _download(cls, archive: str | None = None, timeout: int = 60) -> None:
        url = f"{cls.base_url}/api/access/datafile/{cls.file_id}"

        with requests.get(url, stream=True, timeout=timeout, allow_redirects=True) as response:
            response.raise_for_status()
            archive = cls._archive_name_from_response(response, archive)
            cls.archive = archive
            total = int(response.headers.get("Content-Length", 0))
            desc = os.path.basename(archive)
            with utils.ProgressBar(total=total, unit="B", unit_scale=True, miniters=1, desc=desc) as progress:
                with open(archive, "wb") as file:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if not chunk:
                            continue
                        file.write(chunk)
                        progress.update(len(chunk))

        if not os.path.exists(archive) or os.path.getsize(archive) == 0:
            raise Exception("Download failed.")

    @classmethod
    def _extract(cls, archive: str | None = None) -> None:
        utils.extract_archive(archive or cls.archive, delete=True)

    def create_catalogue(self) -> pd.DataFrame:
        """Creates catalogue for CHIRP dataset."""
        
        assert self.root is not None
        PossibleNeighbour = pd.read_csv(os.path.join(self.root, "ReID","PossibleBirds_Neighbours.csv"))
        PossibleTerritory = pd.read_csv(os.path.join(self.root, "ReID","PossibleBirds_Territory.csv"))

        data = pd.read_csv(os.path.join(self.root,"ReID", "Annotation.csv"))
        df = pd.DataFrame(
            {
                "image_id": data.index,
                "identity": data["id"].astype(str),
                "path": "ReID/" + os.path.sep + data["img"].astype(str).str.replace("/", os.path.sep, regex=False),
                "date": data["Date"],
                "video_name": data["Video"],
                "observation_id": data["UnqTrack"].map({track: idx for idx, track in enumerate(data["UnqTrack"].unique())}),
                "observation_name": data["UnqTrack"],
                "territory": data["Territory"],
                "year": data["Year"],
                "split-closed_set": data["ClosedSetSplit"],
                "split-disjointed_set": data["DisjointedSetSplit"],
                "split-open_set": data["OpenSetSplit"],
                "possible_territory": data["Video"].map(PossibleTerritory.set_index("Video")["PossibleBirds"]),
                "possible_neighbour": data["Video"].map(PossibleNeighbour.set_index("Video")["PossibleBirds"])
            }
        )
        return self.finalize_catalogue(df)
