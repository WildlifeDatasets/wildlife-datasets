import os
import json 
import time
import shutil
import datetime
from typing import List, Optional, Tuple

from datasets import load_dataset
import pandas as pd
from pyinaturalist import get_observations
from . import utils


def check_attribute(obj, attr):
    if not hasattr(obj, attr):
        raise Exception(f'Object {obj} must have attribute {attr}.')


def json_serial(obj):
    if isinstance(obj, (datetime.datetime, datetime.date)):
        return obj.isoformat()
    raise TypeError("Type not serializable")

class DownloadURL:
    url = None
    archive = None
    downloads = []
    rmtree = None
    extract_add_folder = True

    @classmethod
    def _download(cls):
        if cls.url:
            if cls.archive:
                utils.download_url(cls.url, cls.archive)
            else:
                raise ValueError('When cls.url is specified, cls.archive must also be specified')
        for url, archive in cls.downloads:
            utils.download_url(url, archive)        

    @classmethod
    def _extract(cls, exts = ['.zip', '.tar', '.tar.gz', '.tgz', '.tar.bz2', '.rar', '.7z']):
        if cls.archive:
            if any(cls.archive.endswith(ext) for ext in exts):
                utils.extract_archive(cls.archive, delete=True)
        for _, archive in cls.downloads:
            if any(archive.endswith(ext) for ext in exts):
                if cls.extract_add_folder:
                    archive_name = archive.split('.')[0]
                    utils.extract_archive(archive, extract_path=archive_name, delete=True)
                else:
                    utils.extract_archive(archive, delete=True)
        if cls.rmtree:
            shutil.rmtree(cls.rmtree)

class DownloadKaggle:
    @classmethod
    def _download(cls):
        check_attribute(cls, 'kaggle_url')
        check_attribute(cls, 'kaggle_type')
        display_name = cls.display_name().lower()
        if cls.kaggle_type == 'datasets':
            command = f'datasets download -d {cls.kaggle_url} --force'
        elif cls.kaggle_type == 'competitions':
            command = f'competitions download -c {cls.kaggle_url} --force'
        else:
            raise ValueError(f'cls.kaggle_type must be datasets or competitions.')
        exception_text = f'''Kaggle must be setup.
            Check https://wildlifedatasets.github.io/wildlife-datasets/downloads#{display_name}'''
        try:
            os.system(f"kaggle {command}")
        except:
            raise Exception(exception_text)
        if not os.path.exists(cls.archive_name()):
            raise Exception(exception_text)

    @classmethod
    def _extract(cls):
        display_name = cls.display_name().lower()
        try:
            utils.extract_archive(cls.archive_name(), delete=True)
        except:
            exception_text = f'''Extracting failed.
                Either the download was not completed or the Kaggle terms were not agreed with.
                Check https://wildlifedatasets.github.io/wildlife-datasets/downloads#{display_name}'''
            raise Exception(exception_text)
    
    @classmethod
    def archive_name(cls):
        return cls.kaggle_url.split('/')[-1] + '.zip'
        
class DownloadHuggingFace:
    determined_by_df = False
    saved_to_system_folder = True

    @classmethod
    def _download(cls, *args, **kwargs):
        check_attribute(cls, 'hf_url')
        load_dataset(cls.hf_url, *args, **kwargs)

    @classmethod
    def _extract(cls, **kwargs):
        pass


class DownloadINaturalist:
    """
    Download images from iNaturalist for a given user or project.

    Class attributes expected to be set by subclasses:

    - root: directory where images/metadata/csv are stored
    - username: iNaturalist username (optional)
    - project_id: iNaturalist project identifier (optional)
    - target_species_guess: if set, only observations with this species_guess
      are downloaded; if None, all species are accepted.
    """

    username: Optional[str] = None
    project_id: Optional[int | str] = None

    per_page: int = 10
    delay: float = 1.0  # seconds between API pages
    target_species_guess: Optional[List[str]] = None

    # Only a subset of metadata to keep things small
    metadata_fields: Optional[Tuple[str, ...]] = None

    @classmethod
    def _download(cls, force=None):
        if cls.username is None and cls.project_id is None:
            raise ValueError("Provide at least one of username or project_id.")

        page = 1
        csv_records = []

        os.makedirs("images", exist_ok=True)
        os.makedirs("metadata", exist_ok=True)
        while True:
            # Build query parameters depending on what is configured
            params = {
                "per_page": cls.per_page,
                "page": page,
            }
            if cls.username is not None:
                params["user_login"] = cls.username
            if cls.project_id is not None:
                params["project_id"] = cls.project_id

            observations = get_observations(**params)

            if not observations or not observations.get("results"):
                break

            for obs in observations["results"]:
                if cls.target_species_guess is not None:
                    if obs.get("species_guess", "") not in cls.target_species_guess:
                        continue
                
                individual_id = obs.get("id")
                for photo in obs.get("photos", []):
                    photo_id = photo["id"]
                    url_split = photo["url"].split("/")

                    # Use the original extension
                    base_name = url_split[-1]
                    ext = os.path.splitext(base_name)[1]
                    if not ext:
                        ext = ".jpg"

                    url = "/".join(url_split[:-1]) + "/original" + ext

                    file_name_image = f"images/{individual_id}_{photo_id}{ext}"
                    file_name_metadata = f"metadata/{individual_id}_{photo_id}.json"

                    # Skip already downloaded images
                    if not os.path.exists(file_name_image):
                        img = utils.download_image(url, file_name=file_name_image)
                        if not img:
                            print(f'{url}: image download failed')
                            continue

                    if cls.metadata_fields is None:
                        selected_metadata = {
                            key: value for key, value in obs.items() if key != "photos"
                        }
                    else:
                        selected_metadata = {
                            key: obs.get(key) for key in cls.metadata_fields
                        }

                    metadata = {
                        "observation_id": individual_id,
                        "photo_id": photo_id,
                        "photo_url": url,
                        "file_name": file_name_image,
                        **selected_metadata,
                    }

                    with open(file_name_metadata, "w") as f:
                        json.dump(metadata, f, indent=4, default=json_serial)

                    csv_records.append(metadata)

            page += 1
            time.sleep(cls.delay)

        pd.DataFrame(csv_records).to_csv("metadata.csv", index=False)

    @classmethod
    def _extract(cls, *args, **kwargs):
        pass

    @classmethod
    def download(cls, root: str, force: bool = True, **kwargs) -> None:
        """
        Convenience wrapper for ``cls._download`` that updates the dataset.

        Previously downloaded images are skipped to avoid unnecessary transfers,
        while metadata is still checked and refreshed as needed.
        """

        with utils.data_directory(root):
            cls._download(force=force, **kwargs)
        if hasattr(cls, 'summary') and 'licenses_url' in cls.summary and isinstance(cls.summary, str):
            with open(os.path.join(root, cls.license_file_name), 'w') as file:
                file.write(cls.summary['licenses_url'])