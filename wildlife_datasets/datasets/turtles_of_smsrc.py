import os
import json

import pandas as pd
from .downloads import DownloadINaturalist
from .datasets import WildlifeDataset

summary = {
    "licenses": "",
    "licenses_url": "",
    "url": "",
    "publication_url": None,
    "cite": "",
    "animals": {"Green Sea Turtle"},
    "animals_simple": "",
    "real_animals": None,
    "year": None,
    "reported_n_total": None,
    "reported_n_individuals": None,
    "wild": False,
    "clear_photos": True,
    "pose": "",
    "unique_pattern": None,
    "from_video": None,
    "cropped": False,
    "span": "",
    "size": None,
}


class TurtlesOfSMSRC(DownloadINaturalist, WildlifeDataset):
    summary = summary
    username = "jonlape"
    target_species_guess = "Green Sea Turtle"
    metadata_fields = (
        "species_guess",
        "observed_on",
        "location",
    )

    def create_catalogue(self) -> pd.DataFrame:
        records = []
        root = self.root
        valid_exts = self.valid_exts

        for dirpath, _, filenames in os.walk(root):
            for fname in sorted(filenames):
                ext = os.path.splitext(fname)[1].lower()
                if ext not in valid_exts:
                    continue

                img_path = os.path.join(dirpath, fname)
                rel_path = os.path.relpath(img_path, root)

                # Expected metadata file: "<obsid>_<photoid>_metadata.json"
                base_stem = os.path.splitext(img_path)[0]
                meta_path = base_stem + "_metadata.json"

                metadata = {}
                if os.path.exists(meta_path):
                    with open(meta_path, "r") as f:
                        metadata = json.load(f)

                # Try to get IDs from metadata
                observation_id = metadata.get("observation_id")
                photo_id = metadata.get("photo_id")

                # Fallback: parse from filename "<obsid>_<photoid>.<ext>"
                base = os.path.splitext(fname)[0]
                if observation_id is None or photo_id is None:
                    try:
                        obs_id_str, photo_id_str = base.split("_", 1)
                    except ValueError:
                        # If filename doesn't follow the pattern, fallback to using the whole stem
                        obs_id_str = base
                        photo_id_str = base
                    observation_id = observation_id or obs_id_str
                    photo_id = photo_id or photo_id_str

                image_id = f"{observation_id}_{photo_id}"
                identity = os.path.basename(os.path.dirname(img_path))

                record = dict(metadata)

                record.update(
                    {
                        "image_id": image_id,
                        "identity": identity,
                        "path": rel_path,
                    }
                )

                records.append(record)

        if not records:
            raise RuntimeError(
                f"No images found under root={root}. "
                "Did you run TurtlesOfSMSRC.update_data(root) first, and is root correct?"
            )
        df = pd.DataFrame.from_records(records)

        return self.finalize_catalogue(df)
