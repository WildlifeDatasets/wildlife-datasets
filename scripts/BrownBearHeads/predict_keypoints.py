"""Predict BrownBearHeads keypoints with the released BrownBear_ReID HRNet model.

This is optional and heavier than plain metadata preparation. It requires:

- an image root matching the input metadata `path` column,
- the official `amathislab/BrownBear_ReID` repository checkout,
- the repository's pose-model dependencies, especially `torch` and `torchvision`.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import os
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from tqdm import tqdm

CHECKPOINT_NAME = "hrnet_w48_balanced_n13_refined.pth"
DEFAULT_CHECKPOINTS = (
    ("checkpoints", "preprocessing_ckpts", "pose", CHECKPOINT_NAME),
    ("checkpoints", "pose", CHECKPOINT_NAME),
)


def resolve_checkpoint(source_root: str, checkpoint: str | None) -> str:
    """Find the released HRNet pose checkpoint."""

    if checkpoint is not None:
        checkpoint = os.path.expanduser(checkpoint)
        if not os.path.exists(checkpoint):
            raise FileNotFoundError(f"Could not find checkpoint: {checkpoint}")
        return checkpoint

    candidates = [os.path.join(source_root, *parts) for parts in DEFAULT_CHECKPOINTS]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate

    raise FileNotFoundError(f"Could not find `{CHECKPOINT_NAME}`. Use --checkpoint.")


def resolve_image_path(source_root: str, relative_path: str) -> str:
    """Resolve one metadata image path inside the extracted release."""

    relative_path = os.path.expanduser(str(relative_path))
    if os.path.isabs(relative_path) and os.path.exists(relative_path):
        return relative_path

    candidates = [
        os.path.join(source_root, relative_path),
        os.path.join(source_root, "data", relative_path),
        os.path.join(source_root, "images", relative_path),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate

    tried = "\n".join(f"- {candidate}" for candidate in candidates)
    raise FileNotFoundError(f"Could not find image `{relative_path}`. Tried:\n{tried}")


def import_pose_code(repo_root: str):
    """Import official BrownBear_ReID pose utilities without training dependencies."""

    repo_root = os.path.expanduser(repo_root)
    backbones_root = os.path.join(repo_root, "PoseGuidedReID", "project", "models", "backbones")
    pose_net_path = os.path.join(backbones_root, "pose_net.py")
    if not os.path.isfile(pose_net_path):
        raise FileNotFoundError(f"Could not find pose_net.py under: {repo_root}")

    package_hash = hashlib.sha1(os.path.abspath(backbones_root).encode()).hexdigest()[:12]
    package_name = f"_brown_bear_reid_backbones_{package_hash}"
    module_name = f"{package_name}.pose_net"

    if package_name not in sys.modules:
        package = types.ModuleType(package_name)
        package.__path__ = [backbones_root]
        package.__package__ = package_name
        sys.modules[package_name] = package

    if module_name in sys.modules:
        module = sys.modules[module_name]
    else:
        spec = importlib.util.spec_from_file_location(module_name, pose_net_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not import pose_net.py from: {pose_net_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

    return module.SimpleHRNet, module._box2cs, module.get_affine_transform, module.get_final_preds


def choose_device(device: str) -> str:
    """Choose cuda when available, otherwise cpu."""

    if device != "auto":
        return device

    import torch

    return "cuda:0" if torch.cuda.is_available() else "cpu"


def load_rgb_image(path: str) -> np.ndarray:
    """Load one image as an RGB numpy array with EXIF orientation applied."""

    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        return np.asarray(image)


def predict_batch_variable_size(model, images: list[np.ndarray], pose_helpers) -> tuple[np.ndarray, np.ndarray]:
    """Predict keypoints for a list of differently sized RGB images."""

    import cv2
    import torch

    _, box2cs, get_affine_transform, get_final_preds = pose_helpers
    images_tensor = torch.empty(len(images), 3, model.resolution[0], model.resolution[1])
    center_list = []
    scale_list = []

    for i, image in enumerate(images):
        center, scale = box2cs([0, 0, image.shape[1], image.shape[0]], model.resolution[1], model.resolution[0])
        trans = get_affine_transform(center, scale, 0, model.resolution)
        image = cv2.warpAffine(image, trans, model.resolution, flags=cv2.INTER_LINEAR)
        images_tensor[i] = model.transform(image)
        center_list.append(center)
        scale_list.append(scale)

    with torch.no_grad():
        input_device = next(model.model.parameters()).device
        heatmaps, _ = model.model(images_tensor.to(input_device), return_feature=True)
        if isinstance(heatmaps, list):
            heatmaps = heatmaps[-1]
        _, preds, maxvals = get_final_preds(heatmaps.cpu().detach().numpy(), center_list, scale_list)

    return preds, maxvals


def keypoints_out_of_bounds(prediction: np.ndarray, image_shape: tuple[int, ...]) -> np.ndarray:
    """Return a boolean mask for keypoints outside image bounds."""

    height, width = image_shape[:2]
    finite = np.isfinite(prediction).all(axis=1)
    return finite & (
        (prediction[:, 0] < 0) | (prediction[:, 0] >= width) | (prediction[:, 1] < 0) | (prediction[:, 1] >= height)
    )


def flatten_scores(confidence: np.ndarray) -> list[float]:
    """Flatten one [n_keypoints, 1] confidence array."""

    return confidence.astype(float).reshape(-1).tolist()


def flatten_prediction(
    prediction: np.ndarray,
    confidence: np.ndarray,
    confidence_threshold: float,
    image_shape: tuple[int, ...] | None = None,
    drop_out_of_bounds: bool = False,
) -> list[float]:
    """Flatten one [n_keypoints, 2] prediction after optional masking."""

    prediction = prediction.astype(float).copy()
    confidence = confidence.reshape(-1)
    prediction[confidence < confidence_threshold] = np.nan
    if drop_out_of_bounds and image_shape is not None:
        prediction[keypoints_out_of_bounds(prediction, image_shape)] = np.nan
    return prediction.reshape(-1).tolist()


def nullable_int_series(values: list[float]) -> pd.Series:
    """Convert coordinate values to nullable integer CSV columns."""

    return pd.Series(values).map(lambda value: int(round(value)) if np.isfinite(value) else pd.NA).astype("Int64")


def write_head_keypoints_sidecar(metadata: pd.DataFrame, output_csv: str) -> pd.DataFrame:
    """Write pose-only `head_keypoints.csv` keyed by final metadata path."""

    keypoints = [np.asarray(value, dtype=float).reshape(-1, 2) for value in metadata["keypoints"]]
    scores = [np.asarray(value, dtype=float).reshape(-1) for value in metadata["keypoint_scores"]]
    n_keypoints = max([len(points) for points in keypoints] or [0])

    sidecar = pd.DataFrame({"path": metadata["path"].to_numpy()})
    for i in range(n_keypoints):
        sidecar[f"keypoint_{i:02d}_x"] = nullable_int_series(
            [points[i, 0] if i < len(points) else np.nan for points in keypoints]
        )
        sidecar[f"keypoint_{i:02d}_y"] = nullable_int_series(
            [points[i, 1] if i < len(points) else np.nan for points in keypoints]
        )
        sidecar[f"keypoint_{i:02d}_score"] = np.round(
            [score[i] if i < len(score) else np.nan for score in scores],
            6,
        )

    metric_columns = ["min_keypoint_score", "mean_keypoint_score", "n_out_of_bounds_keypoints"]
    sidecar = pd.concat([sidecar, metadata[metric_columns].reset_index(drop=True)], axis=1)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    sidecar.to_csv(output_csv, index=False)
    print(f"Saved {len(sidecar)} rows to {output_csv}", flush=True)
    return sidecar


def predict_keypoints(
    source_root: str,
    metadata_csv: str,
    output_csv: str,
    repo_root: str,
    checkpoint: str | None = None,
    batch_size: int = 32,
    device: str = "auto",
    confidence_threshold: float = 0.0,
    drop_out_of_bounds: bool = False,
    limit: int | None = None,
    sidecar_csv: str | None = None,
) -> pd.DataFrame:
    """Run the released BrownBear_ReID pose model and write keypoints into metadata."""

    source_root = os.path.expanduser(source_root)
    metadata_csv = os.path.expanduser(metadata_csv)
    output_csv = os.path.expanduser(output_csv)
    checkpoint = resolve_checkpoint(source_root, checkpoint)
    pose_helpers = import_pose_code(repo_root)
    simple_hrnet = pose_helpers[0]
    device = choose_device(device)

    print(f"Loading pose model from {checkpoint} on {device}.", flush=True)
    model = simple_hrnet(
        48,
        13,
        checkpoint,
        model_name="HRNet",
        resolution=(256, 256),
        max_batch_size=batch_size,
        device=device,
    )

    metadata = pd.read_csv(metadata_csv, low_memory=False)
    if "path" not in metadata.columns:
        raise ValueError("Input metadata must contain a `path` column.")
    if limit is not None:
        metadata = metadata.head(limit).copy()

    n_batches = (len(metadata) + batch_size - 1) // batch_size
    print(f"Starting inference for {len(metadata)} images in {n_batches} batches.", flush=True)
    predictions = []
    score_predictions = []
    out_of_bounds_counts = []
    min_scores = []
    mean_scores = []
    with tqdm(
        total=len(metadata),
        desc="Predicting keypoints",
        unit="image",
        file=sys.stdout,
        dynamic_ncols=True,
        disable=False,
    ) as progress:
        for batch_index, start in enumerate(range(0, len(metadata), batch_size), start=1):
            rows = metadata.iloc[start : start + batch_size]
            progress.set_postfix(batch=f"{batch_index}/{n_batches}", stage="loading", refresh=True)
            images = [load_rgb_image(resolve_image_path(source_root, path)) for path in rows["path"]]
            progress.set_postfix(batch=f"{batch_index}/{n_batches}", stage="forward", refresh=True)
            coords, scores = predict_batch_variable_size(model, images, pose_helpers)
            progress.set_postfix(batch=f"{batch_index}/{n_batches}", stage="saving", refresh=True)
            for image, coord, score in zip(images, coords, scores, strict=True):
                score_flat = flatten_scores(score)
                finite_scores = np.asarray(score_flat, dtype=float)
                finite_scores = finite_scores[np.isfinite(finite_scores)]
                score_predictions.append(score_flat)
                predictions.append(
                    flatten_prediction(
                        coord,
                        score,
                        confidence_threshold,
                        image_shape=image.shape,
                        drop_out_of_bounds=drop_out_of_bounds,
                    )
                )
                out_of_bounds_counts.append(int(keypoints_out_of_bounds(coord, image.shape).sum()))
                min_scores.append(float(finite_scores.min()) if len(finite_scores) else np.nan)
                mean_scores.append(float(finite_scores.mean()) if len(finite_scores) else np.nan)
            progress.update(len(rows))

    metadata["keypoints"] = pd.Series(predictions, dtype=object)
    metadata["keypoint_scores"] = pd.Series(score_predictions, dtype=object)
    metadata["n_out_of_bounds_keypoints"] = out_of_bounds_counts
    metadata["min_keypoint_score"] = min_scores
    metadata["mean_keypoint_score"] = mean_scores
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    metadata.to_csv(output_csv, index=False)
    print(f"Saved {len(metadata)} rows with predicted keypoints to {output_csv}")
    if sidecar_csv is not None:
        write_head_keypoints_sidecar(metadata, os.path.expanduser(sidecar_csv))
    print(f"Rows with out-of-bounds keypoints: {(metadata['n_out_of_bounds_keypoints'] > 0).sum()}", flush=True)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict BrownBearHeads keypoints with released HRNet checkpoint.")
    parser.add_argument(
        "source_root", help="Image root matching metadata paths, usually the prepared BrownBearHeads root."
    )
    parser.add_argument("metadata_csv", help="Input metadata CSV with a path column.")
    parser.add_argument("output_csv", help="Output metadata CSV with predicted keypoints.")
    parser.add_argument("--repo-root", required=True, help="Path to local amathislab/BrownBear_ReID checkout.")
    parser.add_argument("--checkpoint", default=None, help="Optional explicit HRNet checkpoint path.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--confidence-threshold", type=float, default=0.0)
    parser.add_argument("--drop-out-of-bounds", action="store_true", help="Set out-of-bounds keypoints to NaN.")
    parser.add_argument("--limit", type=int, default=None, help="Optional smoke-test limit.")
    parser.add_argument("--sidecar-csv", default=None, help="Optional pose-only head_keypoints.csv output path.")
    args = parser.parse_args()

    predict_keypoints(
        source_root=args.source_root,
        metadata_csv=args.metadata_csv,
        output_csv=args.output_csv,
        repo_root=args.repo_root,
        checkpoint=args.checkpoint,
        batch_size=args.batch_size,
        device=args.device,
        confidence_threshold=args.confidence_threshold,
        drop_out_of_bounds=args.drop_out_of_bounds,
        limit=args.limit,
        sidecar_csv=args.sidecar_csv,
    )


if __name__ == "__main__":
    main()
