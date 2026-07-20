# BrownBearHeads Preparation

This folder contains the scripts used to build the Kaggle/WildlifeDatasets
version of BrownBearHeads from the Zenodo public release.

The final dataset root should look like this:

```text
BrownBearHeads/
  metadata.csv
  head_keypoints.csv
  BrownBearHeads/
    2017/
      Aardvark/
        598A2507_BLR_0_Aardvark.JPG
    ...
```

`metadata.csv` is the main WildlifeDatasets metadata file. `head_keypoints.csv`
is an optional pose sidecar. Both files must use the same `path` values.

## Prepare Metadata

Create the intermediate metadata from the extracted Zenodo release:

```bash
python scripts/BrownBearHeads/prepare_metadata.py \
  /path/to/Public_release \
  /path/to/BrownBearHeads/clean_metadata.csv
```

This writes one row per image. At this stage `path` and `path_original` both
point to the original Zenodo head-crop path, for example:

```text
2017_heads/images/598A2507_BLR_0_Aardvark.JPG
```

Do not upload `clean_metadata.csv` to Kaggle. It is only an intermediate file.

Sparse Zenodo ground-truth/preprocessing keypoints can be added for inspection:

```bash
python scripts/BrownBearHeads/prepare_metadata.py \
  /path/to/Public_release \
  /path/to/BrownBearHeads/clean_metadata.csv \
  --include-keypoints
```

These sparse keypoints are not the final pose sidecar. The final `metadata.csv`
should not contain pose columns.

## Prepare Images

Create the final image folder and final `metadata.csv`:

```bash
python scripts/BrownBearHeads/resize_and_restructure.py \
  /path/to/Public_release \
  /path/to/BrownBearHeads/clean_metadata.csv \
  /path/to/BrownBearHeads \
  --max-side 720
```

This copies/resizes images and rewrites `path` to the final Kaggle structure:

```text
BrownBearHeads/<year>/<identity>/<filename>
```

The script writes:

```text
/path/to/BrownBearHeads/metadata.csv
```

Final `metadata.csv` has no pose columns. Pose belongs in `head_keypoints.csv`.

## Predict Head Keypoints

Use the released HRNet pose checkpoint from the Zenodo release and a local
checkout of the official `amathislab/BrownBear_ReID` code:

```bash
git clone https://github.com/amathislab/BrownBear_ReID /path/to/BrownBear_ReID
```

Run prediction on the prepared dataset root, not on the raw Zenodo root:

```bash
python scripts/BrownBearHeads/predict_keypoints.py \
  /path/to/BrownBearHeads \
  /path/to/BrownBearHeads/metadata.csv \
  /path/to/BrownBearHeads/metadata_with_keypoints.csv \
  --repo-root /path/to/BrownBear_ReID \
  --checkpoint /path/to/Public_release/checkpoints/preprocessing_ckpts/pose/hrnet_w48_balanced_n13_refined.pth \
  --device cuda:0 \
  --batch-size 32 \
  --sidecar-csv /path/to/BrownBearHeads/head_keypoints.csv
```

`metadata_with_keypoints.csv` is a diagnostic full metadata file with pose
columns. The Kaggle dataset only needs `metadata.csv` and `head_keypoints.csv`.

Smoke test:

```bash
python scripts/BrownBearHeads/predict_keypoints.py \
  /path/to/BrownBearHeads \
  /path/to/BrownBearHeads/metadata.csv \
  /path/to/BrownBearHeads/metadata_with_keypoints_smoke.csv \
  --repo-root /path/to/BrownBear_ReID \
  --checkpoint /path/to/Public_release/checkpoints/preprocessing_ckpts/pose/hrnet_w48_balanced_n13_refined.pth \
  --device cpu \
  --limit 16 \
  --sidecar-csv /path/to/BrownBearHeads/head_keypoints_smoke.csv
```

`head_keypoints.csv` stores pose in wide CSV form:

```text
path,
keypoint_00_x,keypoint_00_y,keypoint_00_score,
...
keypoint_12_x,keypoint_12_y,keypoint_12_score,
min_keypoint_score,mean_keypoint_score,n_out_of_bounds_keypoints
```

Its `path` column must match `metadata.csv["path"]` exactly.

## Repair Old Sidecar

If an older `head_keypoints.csv` has paths like `2017/images/...` or
`2017_heads/images/...`, repair it once:

```bash
python scripts/BrownBearHeads/fix_head_keypoint_paths.py \
  /path/to/BrownBearHeads \
  --dry-run
```

Then overwrite the sidecar:

```bash
python scripts/BrownBearHeads/fix_head_keypoint_paths.py \
  /path/to/BrownBearHeads
```

The repair maps keypoints by `year + filename` and rewrites the sidecar `path`
to the exact final metadata path.

## Load

```python
from wildlife_datasets import datasets

root = "/path/to/BrownBearHeads"
dataset = datasets.BrownBearHeads(root)
dataset_with_pose = datasets.BrownBearHeads(root, load_keypoints=True)
```

`load_keypoints=True` joins `head_keypoints.csv` to `metadata.csv` by `path`
and converts the wide sidecar columns into `keypoints` and `keypoint_scores`.

## Final Metadata Columns

Final `metadata.csv` contains:

- `image_id`
- `identity`
- `path`
- `date`
- `year`
- `camera`
- `split_2017`
- `split_2018`
- `split_2019`
- `split_2020`
- `split_2021`
- `split_2022`
- `split_ood`
- `split_iid`
- `width`
- `height`
- `width_original`
- `height_original`
- `species` if added before upload

Final `metadata.csv` should not contain:

- `path_original`
- `keypoints`
- `keypoint_scores`
- `min_keypoint_score`
- `mean_keypoint_score`
- `n_out_of_bounds_keypoints`

## Splits

The original yearly columns `split_2017` to `split_2022` preserve the released
year-specific protocols. Original `train_iid`, `val_iid`, `test_iid`, and
`test_ood` are normalized to `train`, `val`, or `test`.

`split_ood` is the standardized out-of-distribution split:

- years 2017-2020: `train`
- year 2021: `val`
- year 2022: `test`

`split_iid` is an identity-wise chronological split by observation day:

- earliest 60% of days: `train`
- next 10% of days: `val`
- latest 30% of days: `test`
