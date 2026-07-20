"""Prepare BrownBearHeads metadata from the extracted Zenodo release."""

from __future__ import annotations

import argparse
import os

import pandas as pd

try:
    from .metadata_cleanup import main as prepare_metadata
except ImportError:  # pragma: no cover - allows running as a plain script.
    from metadata_cleanup import main as prepare_metadata


def main(source_root: str, output_csv: str, include_keypoints: bool = False) -> pd.DataFrame:
    """Write intermediate metadata with original Zenodo image paths."""

    return prepare_metadata(
        root=os.path.expanduser(source_root),
        output_csv=os.path.expanduser(output_csv),
        include_keypoints=include_keypoints,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare BrownBearHeads metadata from Zenodo files.")
    parser.add_argument("source_root", help="Path to the extracted Zenodo Public_release folder.")
    parser.add_argument("output_csv", help="Output metadata CSV path.")
    parser.add_argument(
        "--include-keypoints",
        action="store_true",
        help="Attach sparse Zenodo ground-truth keypoints to intermediate metadata.",
    )
    args = parser.parse_args()

    main(args.source_root, args.output_csv, include_keypoints=args.include_keypoints)
