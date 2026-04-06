# BrownBearHeads Standardization

## Why this exists

The original BrownBearHeads release form is not aligned with what is expected by the `wildlife-datasets`, e.g. one metadata file, one canonical image copy per image, one simple folder structure, and explicitly named split columns.

Mo precisely, the official public release of the dataset 
- includes everything (even including model checkpoints etc.) in one zip file which makes it much harder to download due to the size, i.e., 30Gb.
- stores multiple experiment definitions separately, for example `test_on_2017`, `test_on_2018`, ..., `test_on_2022`, each with their own `train_iid.csv`, `val_iid.csv`, `test_iid.csv`, and `test_ood.csv`.

The goal of this workflow is to convert the release into a **clean, reproducible, standardized local dataset** that can later be loaded in the same spirit as other datasets available through `wildlife-datasets`.


## What the workflow produces

The pipeline creates:
- one cleaned metadata table derived from the original CSV files,
- one canonical image folder with no repeated image files,

The standardized prepared folder is intended to be easy to:
- inspect manually,
- load in Python,
- version and reproduce,
- integrate with `wildlife-datasets`.


## The two steps

The workflow is intentionally split into two scripts so that each script has one clear task.

### 1. `scripts/BrownBearHears/metadata_cleanup.py`

This is the **CSV-only** step.

It:
- loads all available split CSV files from the original BrownBearHeads release,
- merges repeated rows describing the same original image path,
- normalizes dates, years, and selected metadata fields,
- creates explicit split columns for the standardized dataset.

### 2. `scripts/BrownBearHears/brown_bear_heads_resize_images.py`

This is the **image preparation** step.

It:
- reads the cleaned metadata created by `metadata_cleanup.py`,
- resolves each original CSV image path to a real file in the raw release,
- copies and optionally resizes that image,
- writes the final prepared `metadata.csv`.



## Example Usage

```bash
python scripts/BrownBearHears/metadata_cleanup.py \
  /path/to/Public_release \
  /path/to/output/metadata_clean.csv
```

```bash
python scripts/BrownBearHears/brown_bear_heads_resize_images.py \
  /path/to/Public_release \
  /path/to/metadata_clean.csv \
  /path/to/output_root \
  --max-side 720 \
  --workers 8
```

```bash
python scripts/BrownBearHears/brown_bear_heads_build_prepared.py \
  /path/to/Public_release \
  /path/to/output_root \
  --max-side 720 \
  --workers 8
```


## Output structure

The prepared output has this structure:

```text
Public_release_720_prepared/
  metadata_clean.csv
  metadata.csv
  preparation_config.csv
  images/
    2017/
      IdentityA/
        file.jpg
    2018/
      IdentityB/
        file.jpg
```

## Standardized split columns

The cleaned metadata step creates 8 explicit split columns: `split_2017`, `split_2018`, `split_2019`, `split_2020`, `split_2021`, `split_2022`, `split_ood`, `split_iid`

### Original yearly experiment columns

The columns `split_2017` to `split_2022` come from the original release structure.
F or a given image, each of these columns records the role that the image plays in the corresponding original experiment:
`train`, `val`, `test`, empty / missing if that image does not appear in that experiment

To make the representation simpler and more consistent:

- `train_iid.csv` becomes `train`
- `val_iid.csv` becomes `val`
- `test_iid.csv` becomes `test`
- `test_ood.csv` becomes `test`

This keeps the standardized output compact while still preserving experiment membership.


### Derived canonical split: `split_ood`

`split_ood` is a **new standardized split**, not a direct copy of one original CSV.
Its purpose is to provide one simple out-of-distribution protocol that can be used consistently with `wildlife-datasets`.  Definition:

- all images from year `2022` are assigned to `test`,
- all images from year `2021` are assigned to `val`,
- all images from years `2017` to `2020` are assigned to `train`.


### Derived canonical split: `split_iid`

`split_iid` is another **new standardized split**, also not a direct copy of one original CSV.
Its purpose is to provide one simple in-distribution protocol that can be reproduced and compared across methods.
Definition:

- for each identity, observations are sorted chronologically,
- the earliest `60%` are assigned to `train`,
- the next `10%` are assigned to `val`,
- the latest `30%` are assigned to `test`.

This means the split is day-based rather than image-based. That is important because it avoids leakage where near-duplicate images from the same day would appear in both training and testing.


## Final metadata columns

The final prepared `metadata.csv` contains:

- `identity`
- `path`
- `width`
- `height`
- `width_original`
- `height_original`
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