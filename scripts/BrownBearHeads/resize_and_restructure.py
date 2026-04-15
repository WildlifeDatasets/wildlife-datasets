"""Resize BrownBearHeads images and build the year/identity oriented folder structure.

This script reads the merged metadata created by `metadata_cleanup.py`, resolves each
original image path to a real file inside the raw release, resizes or copies
the image, stores it under `BrownBearHeads/<year>/<identity>/`, and writes the final
prepared `metadata.csv`.
"""

import argparse
import hashlib
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from PIL import Image, ImageOps
from tqdm import tqdm


def load_metadata(metadata_path: str) -> pd.DataFrame:
    """Load the merged metadata CSV."""

    return pd.read_csv(metadata_path, low_memory=False)


def sanitize_component(value) -> str:
    """Make one path component safe for use in the output folder structure."""

    text = str(value).strip()
    if not text:
        return "Unknown"

    safe = []
    for char in text:
        if char.isalnum() or char in {"-", "_"}:
            safe.append(char)
        else:
            safe.append("_")

    sanitized = "".join(safe).strip("_")
    return sanitized or "Unknown"


def resize_keep_aspect(image: Image.Image, max_side: int) -> Image.Image:
    """Resize an image so its longest side is at most `max_side`."""

    width, height = image.size
    if max(width, height) <= max_side:
        return image.copy()

    scale = max_side / max(width, height)
    new_size = (round(width * scale), round(height * scale))
    return image.resize(new_size, Image.Resampling.BICUBIC)


def process_one_image(source_path: str, output_path: str, max_side: int) -> tuple[int, int, int, int]:
    """Resize or copy one image and return new and original dimensions."""

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with Image.open(source_path) as image:
        image = ImageOps.exif_transpose(image)
        if image.mode not in {"RGB", "L"}:
            image = image.convert("RGB")
        width_original, height_original = image.size
        resized = resize_keep_aspect(image, max_side=max_side)
        width, height = resized.size

        save_kwargs = {}
        if Path(output_path).suffix.lower() in {".jpg", ".jpeg"}:
            save_kwargs = {"quality": 90}
        resized.save(output_path, **save_kwargs)

    return width, height, width_original, height_original


def build_target_path(source_relpath: str, year, identity, used_paths: set[str]) -> str:
    """Create one canonical output path under `BrownBearHeads/<year>/<identity>/`."""

    year_text = sanitize_component(year)
    identity_text = sanitize_component(identity)
    filename = os.path.basename(source_relpath)

    target_path = os.path.join("BrownBearHeads", year_text, identity_text, filename)
    if target_path in used_paths:
        stem, suffix = os.path.splitext(filename)
        digest = hashlib.sha1(source_relpath.encode("utf-8")).hexdigest()[:8]
        target_path = os.path.join("BrownBearHeads", year_text, identity_text, f"{stem}_{digest}{suffix}")

    used_paths.add(target_path)
    return target_path


def resize_and_restructure(
    metadata: pd.DataFrame,
    source_root: str,
    prepared_root: str,
    max_side: int,
    workers: int,
) -> pd.DataFrame:
    """Resize and restructure the dataset in one metadata pass."""

    used_paths: set[str] = set()
    prepared_records = []
    columns = metadata.columns.tolist()

    with ProcessPoolExecutor(max_workers=workers) as executor:
        future_to_record = {}

        for values in metadata.itertuples(index=False, name=None):
            record = dict(zip(columns, values))
            source_relpath = os.path.normpath(str(record.pop("path_original")))
            source_path = os.path.join(source_root, source_relpath)
            if not os.path.exists(source_path):
                raise FileNotFoundError(f"Could not find source image: {source_path}")

            target_path = build_target_path(
                source_relpath=source_relpath,
                year=record["year"],
                identity=record["identity"],
                used_paths=used_paths,
            )
            output_path = os.path.join(prepared_root, target_path)

            future = executor.submit(process_one_image, source_path, output_path, max_side)
            future_to_record[future] = (record, target_path)

        for future in tqdm(as_completed(future_to_record), total=len(future_to_record), desc="Processing images"):
            record, target_path = future_to_record[future]
            width, height, width_original, height_original = future.result()
            record["path"] = target_path
            record["width"] = width
            record["height"] = height
            record["width_original"] = width_original
            record["height_original"] = height_original
            prepared_records.append(record)

    return pd.DataFrame(prepared_records)


def main(
    source_root: str = "",
    metadata_path: str = "",
    prepared_root: str = "",
    max_side: int = 720,
    workers: int = max(1, (os.cpu_count() or 1) - 1),
) -> pd.DataFrame:
    """Run the resize and restructure step and write `metadata.csv`."""

    Path(prepared_root).mkdir(parents=True, exist_ok=True)
    metadata = load_metadata(metadata_path)
    prepared = resize_and_restructure(
        metadata,
        source_root=source_root,
        prepared_root=prepared_root,
        max_side=max_side,
        workers=workers,
    )

    output_metadata_path = os.path.join(prepared_root, "metadata.csv")
    prepared.to_csv(output_metadata_path, index=False)
    print(f"Saved {len(prepared)} rows to {output_metadata_path}")
    return prepared


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resize and restructure BrownBearHeads into the prepared layout.")
    parser.add_argument("source_root", nargs="?", default="~/Public_release")
    parser.add_argument(
        "metadata_path",
        nargs="?",
        default="~/BrownBearHears/clean_metadata.csv",
    )
    parser.add_argument("prepared_root", nargs="?", default="~/BrownBearHears")
    parser.add_argument("--max-side", type=int, default=720)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 1) - 1))
    args = parser.parse_args()

    main(
        source_root=args.source_root,
        metadata_path=args.metadata_path,
        prepared_root=args.prepared_root,
        max_side=args.max_side,
        workers=args.workers,
    )
