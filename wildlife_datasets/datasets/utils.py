import ast
import hashlib
import io
import os
import shutil
import urllib.request
from contextlib import contextmanager

import cv2
import numpy as np
import pandas as pd
import requests
from PIL import Image, ImageOps
from tqdm import tqdm


def load_image(path: str, max_size: int | None = None) -> Image.Image:
    """Loads an image.

    Args:
        path (str): Path of the image.
        max_size (Optional[int], optional): Maximal size of the image or None (no restriction).

    Returns:
        Loaded image.
    """

    # We load it with OpenCV because PIL does not apply metadata.
    img = cv2.imread(path)
    if img is None:
        # Fall back to other loading (sometimes works for non-ASCII files).
        try:
            data = np.fromfile(path, dtype=np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        except Exception:
            img = None
        if img is None:
            raise ValueError(f"Image was not loaded properly: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    if max_size is not None:
        w, h = img.size
        if max(w, h) > max_size:
            c = max_size / max(w, h)
            img = img.resize((int(c * w), int(c * h)))
    return img


def get_image(*args, **kwargs) -> Image.Image:
    print("This function will be removed in future releases. Use load_image() instead.")
    return load_image(*args, **kwargs)


def crop_black(img: Image.Image) -> Image.Image:
    """Crops black borders from an image.

    Args:
        img (Image): Image to be cropped.

    Returns:
        Cropped image.
    """

    y_nonzero, x_nonzero, _ = np.nonzero(np.asarray(img))
    return img.crop(
        (
            int(np.min(x_nonzero)),
            int(np.min(y_nonzero)),
            int(np.max(x_nonzero)),
            int(np.max(y_nonzero)),
        )
    )


def crop_white(img: Image.Image) -> Image.Image:
    """Crops white borders from an image.

    Args:
        img (Image): Image to be cropped.

    Returns:
        Cropped image.
    """

    y_nonzero, x_nonzero, _ = np.nonzero(np.asarray(ImageOps.invert(img)))
    return img.crop(
        (
            int(np.min(x_nonzero)),
            int(np.min(y_nonzero)),
            int(np.max(x_nonzero)),
            int(np.max(y_nonzero)),
        )
    )


def find_images(root: str, img_extensions: tuple[str, ...] = (".png", ".jpg", ".jpeg")) -> pd.DataFrame:
    """Finds all image files in folder and subfolders.

    Args:
        root (str): The root folder where to look for images.
        img_extensions (Tuple[str, ...], optional): Image extensions to look for, by default ('.png', '.jpg', '.jpeg').

    Returns:
        Dataframe of relative paths of the images.
    """

    # TODO: does not handle heic (CatIndividualImages) and webp images (ReunionTurtles)
    data = []
    for path, directories, files in os.walk(root):
        for file in files:
            if file.lower().endswith(tuple(img_extensions)):
                data.append({"path": os.path.relpath(path, start=root), "file": file})
    return pd.DataFrame(data)


def find_file_types(root: str) -> pd.Series:
    """Finds all counted file extensions in lowercase in folder and subfolders.

    Args:
        root (str): The root folder where to look for data.

    Returns:
        Dataframe of counts of the extensions.
    """

    data = []
    for path, directories, files in os.walk(root):
        for file in files:
            extension = os.path.splitext(file.lower())[1]
            data.append({"extension": extension})
    return pd.DataFrame(data).value_counts()


def create_id(string_col: pd.Series) -> pd.Series:
    """Creates unique ids from string based on MD5 hash.

    Args:
        string_col (pd.Series): List of ids.

    Returns:
        List of encoded ids.
    """

    entity_id = string_col.apply(lambda x: hashlib.md5(x.encode()).hexdigest()[:16])
    assert len(entity_id.unique()) == len(entity_id)
    return entity_id


def bbox_segmentation(bbox: list[float]) -> list[float]:
    """Convert bounding box to segmentation.

    Args:
        bbox (List[float]): Bounding box in the form [x, y, w, h].

    Returns:
        Segmentation mask in the form [x1, y1, x2, y2, ...].
    """

    return [
        bbox[0],
        bbox[1],
        bbox[0] + bbox[2],
        bbox[1],
        bbox[0] + bbox[2],
        bbox[1] + bbox[3],
        bbox[0],
        bbox[1] + bbox[3],
        bbox[0],
        bbox[1],
    ]


def segmentation_bbox(segmentation: list[float]) -> list[float]:
    """Convert segmentation to bounding box.

    Args:
        segmentation (List[float]): Segmentation mask in the form [x1, y1, x2, y2, ...].

    Returns:
        Bounding box in the form [x, y, w, h].
    """

    x = segmentation[0::2]
    y = segmentation[1::2]
    x_min = np.min(x)
    x_max = np.max(x)
    y_min = np.min(y)
    y_max = np.max(y)
    return [x_min, y_min, x_max - x_min, y_max - y_min]


def is_annotation_bbox(segmentation: list[float], bbox: list[float], tol: float = 0) -> bool:
    """Checks whether segmentation is bounding box.

    Args:
        segmentation (List[float]): Segmentation mask in the form [x1, y1, x2, y2, ...].
        bbox (List[float]): Bounding box in the form [x, y, w, h].
        tol (float, optional): Tolerance for difference.

    Returns:
        True if segmentation is bounding box within tolerance.
    """

    bbox_seg = bbox_segmentation(bbox)
    if len(segmentation) == len(bbox_seg):
        for x, y in zip(segmentation, bbox_seg):
            if abs(x - y) > tol:
                return False
    else:
        return False
    return True


class ProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path="."):
    with ProgressBar(unit="B", unit_scale=True, miniters=1, desc=url.split("/")[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


def extract_archive(archive, extract_path=".", delete=False):
    shutil.unpack_archive(archive, extract_path)
    if delete:
        os.remove(archive)


@contextmanager
def data_directory(dir):
    """
    Changes context such that data directory is used as current work directory.
    Data directory is created if it does not exist.
    """
    current_dir = os.getcwd()
    if not os.path.exists(dir):
        os.makedirs(dir)
    os.chdir(dir)
    try:
        yield
    finally:
        os.chdir(current_dir)


def gdown_download(url, archive, exception_text=""):
    import gdown

    gdown.download(url, archive, quiet=False)
    if not os.path.exists(archive):
        print(exception_text)
        raise Exception(exception_text)


def get_split(x, data_train, data_test):
    if x in data_train:
        return "train"
    elif x in data_test:
        return "test"
    else:
        return np.nan


def get_image_date(path, shorten=True):
    try:
        exif = Image.open(path).getexif()
    except Exception:
        return -1

    def clean_date(d):
        if isinstance(d, bytes):
            d = d.decode(errors="ignore")
        if len(d) >= 10 and shorten:
            d = d[:10].replace(":", "-")
        return d

    if exif:
        for tag in (36867, 36868, 306):
            date = exif.get(tag)
            if date:
                return clean_date(date)
    return np.nan


def yolo_to_pascalvoc(x_c, y_c, w, h, W, H):
    x_min = int((x_c - w / 2) * W)
    y_min = int((y_c - h / 2) * H)
    x_max = int((x_c + w / 2) * W)
    y_max = int((y_c + h / 2) * H)
    return x_min, y_min, x_max, y_max


def download_image(url, headers=None, file_name=None):
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        img = Image.open(io.BytesIO(response.content))
        if file_name is not None:
            img.save(file_name)
        return img
    elif response.status_code == 404:
        print(f"Image not found (404). Skipping... {url}")
    else:
        print(f"Failed to download image with status code {response.status_code}. {url}")
        try:
            message = response.content.decode("utf-8")
            message = message.split("<Details>")[1]
            message = message.split("</Details>")[0]
            print(message)
        except Exception:
            pass
    return None


def parse_bbox_mask(x):
    if not isinstance(x, str):
        return x
    try:
        return ast.literal_eval(x)
    except (ValueError, SyntaxError):
        raise ValueError(f"Invalid bbox or mask value: {x}")


def strip_suffixes(x: str, suffixes: list[str]) -> str:
    suffixes = suffixes + [x.lower() for x in suffixes]
    while any(x.lower().endswith(ext) for ext in suffixes):
        for ext in suffixes:
            if x.lower().endswith(ext):
                x = x[: -len(ext)].strip()
                break
    return x


def get_persistent_id(paths: pd.Series) -> pd.Series:
    folders = paths.str.split(os.path.sep, expand=True)
    counts = folders.nunique()
    varying = counts[counts > 1]
    if varying.empty:
        raise ValueError("All entries are the same")
    idx = varying.index.min()
    return create_id(paths.str.split(os.path.sep).apply(lambda x: "/".join(x[idx:])))


def load_segmentation(metadata: pd.DataFrame, file_name: str) -> pd.DataFrame:
    # Load segmentation
    segmentation = pd.read_csv(file_name)

    # Merge metadata and segmentation (may result in nans in segmentations)
    cols = ["bbox_x", "bbox_y", "bbox_w", "bbox_h"]
    metadata = pd.merge(metadata, segmentation, on="image_id", how="left")
    metadata["bbox"] = metadata[cols].to_numpy().tolist()

    # Check that there is no image_id with two nans
    mask = metadata[cols].isnull().all(axis=1)
    max_n_image_id = metadata.loc[mask, "image_id"].value_counts().max()
    if max_n_image_id > 1:
        raise ValueError("There is image_id with multiple nan bounding boxes")

    # Generate new image_id
    cols_enhanced = ["image_id"] + cols
    new_image_id = metadata[cols_enhanced].round(2).astype(str).agg("_".join, axis=1)
    new_image_id = get_persistent_id(new_image_id)
    metadata["image_id"] = metadata["image_id"].astype(str) + "_" + new_image_id

    # Finalize the dataframe
    metadata = metadata.drop(cols, axis=1)
    metadata = metadata.reset_index(drop=True)
    return metadata


def find_corrupted_images(root: str) -> list[str]:
    images = find_images(root)
    names = images["path"].str.cat(images["file"], sep=os.path.sep)
    corrupted = []
    for name in tqdm(names):
        try:
            load_image(os.path.join(root, name))
        except Exception:
            corrupted.append(name)
    return corrupted


def delete_corrupted_images(
    root: str, corrupted: list[str], img_extensions: tuple[str, ...] = (".png", ".jpg", ".jpeg")
) -> None:

    for name in corrupted:
        full_name = os.path.join(root, name)
        if os.path.exists(full_name) and name.lower().endswith(img_extensions):
            try:
                load_image(full_name)
                print(f"File is not corrupted: {full_name}")
            except ValueError:
                os.remove(full_name)
