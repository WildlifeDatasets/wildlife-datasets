Each dataset is represented by a pandas dataframe. Each row corresponds to one entry, which depicts one individual animal. Usually one entry corresponds to one image and one animal. However, sometimes, there are multiple animals in an image and then one image may generate multiple entries differentiated by additional information such as bounding box. Columns are descriptions of the entry.


## Required columns

The following three columns must be part of all dataframes.

| Column | Type | Description |
|--------|------|-------------|
| id | `int` or `str` | Unique id of the entry. |
| identity | `int` or `str` | Identity (or label) of the depicted individual animal. |
| path | `str` | Relative path to the image. |
| split | `str` | Automatically generated. Determines whether the sample belongs to the training or testing set. It is `null` whenever the animal identity is unknown. |

There is a special value for `identity` which describes an unknown individual. Its default value for unknown animals is

```python exec="true" source="above" result="console"
from wildlife_datasets import datasets

datasets.DatasetFactory.unknown_name
print(datasets.DatasetFactory.unknown_name) # markdown-exec: hide
```

When a dataset contains unknown inidividuals, there identities should be changed to the default value.


## Optional columns

The following columns may be present in the dataframe. Besides these columns, it is possible to define additional columns.

| Column | Type | Description |
|--------|------|-------------|
| bbox | `List[float]` | Bounding box in the form [x, y, w, h]. Therefore, the topleft corner has coordinates [x, y], while the bottomright corner has coordinates [x+w, y+h]. |
| date | special | Timestamp of the photo. The preferred format is `%Y-%m-%d %H:%M:%S` from the `datetime` package but it is sufficient to be amenable to `pd.to_datetime(x)`. |
| keypoints | `List[float]` | Keypoints coordinates in the image such as eyes or joints. |
| position | `str` | Position from each the photo was taken. The usual values are left and right. |
| segmentation | `List[float]` or special | Segmentation mask in the form [x1, y1, x2, y2, ...]. Additional format are possible such as file path to a mask image, or `pytorch` RLE. |
| species | `str` or `List[str]` | The depicted species for datasets with multiple species. |
| video | `int` | The index of a video. |


## Metadata

Besides the dataframe, each dataset also contains some metadata. The metadata are saved in a separate csv file, which currently contains the following information. All entries are optional.

| Column | Description |
|--------|-------------|
| name | Name of the dataset. |
| licenses | License file for the dataset. |
| licenses_url | URL for the license file. |
| url | URL for the dataset. |
| cite | Citation in Google Scholar type of the paper. |
| animals | List of all animal species in the dataset. |
| real_animals | Determines whether the dataset contains real animals. |
| reported_n_total | The reported number of total animals. |
| reported_n_identified | The reported number of identified animals. |
| reported_n_photos | The reported number of photos. |
| wild | Determines whether the photos were taken in the wild. |
| clear_photos | Determines whether the photos are clear. |
| pose | Determines whether the photos have one orientation (single), two orientation such as left and right flanks (double) or more (multiple). |
| unique_pattern | Determines whether the animals have unique features (fur patern, fin shape) for recognition. |
| from_video | Determines whether the dataset was created from photos or videos. |
| cropped | Determines whether the photos are cropped. |
| span | The span of the dataset (the time difference between the last and first photos). |
