Each dataset is represented by a pandas dataframe. Each row corresponds to one entry, which depicts one individual animal. Usually one entry corresponds to one image and one animal. However, sometimes, there are multiple animals in an image and then one image may generate multiple entries differentiated by additional information such as bounding box. Columns are descriptions of the entry.


## Required columns

The following three columns must be part of all dataframes.

| Column   | Type | Description |
|----------|------|-------------|
| image_id | `int` or `str` | Unique id of the entry. |
| identity | `int` or `str` | Identity (or label) of the depicted individual animal. |
| path     | `str` | Relative path to the image. |

There is a special value for `identity` which describes an unknown individual. Its default value for unknown animals is

```python exec="true" source="above" result="console"
from wildlife_datasets import datasets

datasets.WildlifeDataset.unknown_name
print(datasets.WildlifeDataset.unknown_name) # markdown-exec: hide
```

When a dataset contains unknown individuals, the identity entry should be changed to the default value described above.


## Bounding boxes and segmentations

The following two columns are optional but they are handled specially by the library.

| Column | Type | Description |
|--------|------|-------------|
| bbox | `List[float]` | Bounding box in the form [x, y, w, h]. Therefore, the topleft corner has coordinates [x, y], while the bottomright corner has coordinates [x+w, y+h]. |
| segmentation | `List[float]` or special | Segmentation mask in the form [x1, y1, x2, y2, ...]. Additional formats are possible such as a file path to a mask image or the COCO RLE format `{"size": [h, w], "counts": ...}`. |

The `img_load` keyword uses these columns to crop or mask the images on the fly, without creating new image files. The most important options are `bbox` (crop to the bounding box) and `bbox_mask` (crop to the bounding box and black out the background). See [loading bounding boxes](wildlife_dataset.md#loading-bounding-boxes) for more examples.


## Optional columns

The following columns may be present in the dataframe. Besides these columns, it is possible to define additional dataset-specific columns. The [available metadata](metadata.md) page shows which columns are available for each dataset.

| Column | Type | Description |
|--------|------|-------------|
| date | `str` | Timestamp of the photo. The preferred format is `%Y-%m-%d %H:%M:%S` (or `%Y-%m-%d` when the time is unknown) but it is sufficient to be amenable to `pd.to_datetime(x)`. |
| split | `str` | Split provided by the dataset authors. The usual values are `train`, `val` and `test`, or `query` and `gallery` for re-identification splits. Datasets with multiple splits use additional columns starting with `split_`. |
| orientation | `str` | Side or direction from which the animal was photographed. The usual values are `left`, `right`, `front`, `back`, `top` and `below` and their combinations such as `frontleft`. |
| species | `str` | The depicted species. |
| camera | `str` | Identifier of the camera (camera trap) which took the photo. |
| keypoints | `dict[str, tuple]` | Mapping from keypoint name (e.g. `left_eye`, `nose`) to its coordinates `(x, y)`, optionally followed by extra per-point values such as a detection confidence or a COCO-style visibility flag (0: not labeled, 1: labeled but not visible, 2: visible), e.g. `(x, y, confidence)`. A keypoint is omitted from the dict rather than kept with missing coordinates. |
| video | `int` or `str` | Identifier (index or name) of the video from which the image was extracted. |
| frame_number | `int` | Index of the frame in the video. |
| encounter_id | `int` or `str` | Identifier of an encounter (sighting or tracklet). All images with the same `encounter_id` depict the same individual during one encounter. |
| image_quality | `int` | Quality of the image, where 1 is the best quality and higher values are worse. The scale depends on the dataset. |
| gender | `str` | Sex of the depicted individual. |
| latitude | `float` | Latitude of the location where the photo was taken. |
| longitude | `float` | Longitude of the location where the photo was taken. |
| location | `str` | Name or code of the location (site, territory) where the photo was taken. |
| age | `float` | Age of the depicted individual in years. |
| age_group | `str` | Age category of the depicted individual such as `Juvenile` or `Adult`. |


## Metadata

Besides the dataframe, each dataset also contains some metadata. All entries are optional.

| Column | Description |
|--------|-------------|
| licenses | License file for the dataset. |
| licenses_url | URL for the license file. |
| url | URL for the dataset. |
| cite | Citation in Google Scholar type of the paper. |
| animals | List of all animal scientific names in the dataset. |
| animals_simple | List of all animal common names in the dataset. |
| real_animals | Determines whether the images are of real animals as opposed to computer generated image. |
| year | Publication year of the dataset. |
| reported_n_total | The reported number of total animals. |
| reported_n_individuals | The reported number of individuals. |
| wild | Determines whether the environment that it was photographed in is wild. |
| clear_photos | Determines whether the database is quality-controlled such that image quality is consistent across all images. |
| pose | Determines whether the photos have one orientation (single), two orientation such as left and right flanks (double) or more (multiple). |
| unique_pattern | Determines whether the animals have unique features (fur patern, fin shape) for recognition. |
| from_video | Determines whether the dataset was created from photos or videos. |
| cropped | Determines whether the photos are cropped. |
| span | The span of the dataset (the time difference between the last and first photos). |
| size | Size of the zipped datasets (in MB). |
