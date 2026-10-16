# Available metadata

The table below shows which [optional columns](dataframe.md#optional-columns) are available for each dataset. A checkmark ✓ means that the column is filled for all images, a percentage means that it is filled only for a part of the images, and an empty cell means that the column is not present. Dataset-specific columns are not listed. MetaDatasets and segmented versions of datasets are omitted.

<div class="metadata-table" markdown>

```python exec="true"
import pandas as pd

df = pd.read_csv("docs/csv/metadata.csv", index_col="dataset")
footnotes = {
    ("BrownBearHeads", "keypoints"): 1,
    ("CHIRP", "keypoints"): 1,
    ("Melops", "keypoints"): 1,
    ("Melops", "bbox"): 2,
}

def format_cell(dataset, column, value):
    if pd.isna(value):
        return ""
    text = "✓" if value == 1 else (f"{int(100 * value)}%" if value >= 0.01 else "<1%")
    if (dataset, column) in footnotes:
        text += f"<sup>{footnotes[dataset, column]}</sup>"
    return text

print("| Dataset | " + " | ".join(df.columns) + " |")
print("|:--|" + "|".join(":-:" for _ in df.columns) + "|")
for dataset, row in df.iterrows():
    cells = [format_cell(dataset, column, value) for column, value in row.items()]
    print(f"| {dataset} | " + " | ".join(cells) + " |")
print("| **Total** | " + " | ".join(f"**{n}**" for n in df.notna().sum()) + " |")
```

</div>

<sup>1</sup> Requires `load_keypoints=True` when creating the dataset, for example `datasets.CHIRP(root, load_keypoints=True)`.<br>
<sup>2</sup> Requires the `bbox` argument when creating the dataset, for example `datasets.Melops(root, bbox="head")`.
