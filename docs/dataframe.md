Each dataset is represented by a pandas dataframe. Each row corresponds to one entry, which depicts one individual animal. Usually one entry corresponds to one image and one animal. However, sometimes, there are multiple animals in an image and then one image may generate multiple entries differentiated by additional information such as bounding box. Columns are descriptions of the entry.


## Required columns

The following three columns must be part of all dataframes.

| Column | Type | Description |
|--------|------|-------------|
| id | `int` or `str` | Unique id of the entry. |
| identity | `int` or `str` | Identity (or label) of the depicted individual animal. |
| path | `str` | Relative path to the image. |


## Optional columns

The following columns may be present in the dataframe. Besides these columns, it is possible to define additional columns.

| Column | Type | Description |
|--------|------|-------------|
| bbox | `List[float]` | Bounding box in the form [x, y, w, h]. Therefore, the topleft corner has coordinates [x, y], while the bottomright corner has coordinates [x+w, y+h]. |
| date | special | Timestamp of the photo. The preferable format is `%Y-%m-%d %H:%M:%S` from the `datetime` package but it is sufficient to be amenable to `pd.to_datetime(x)`. |
| keypoints | `List[float]` | Keypoints coordinates in the image such as eyes or joints. |
| position | `str` | Position from each the photo was taken. The usual values are left and right. |
| segmentation | `List[float]` or special | Segmentation mask in the form [x1, y1, x2, y2, ...]. Additional format are possible such as file path to a mask image, or `pytorch` RLE. |
| species | `str` or `List[str]` | The depicted species for datasets with multiple species. |


<!---
TODO: add date to the list

TODO: do something about the other arguments

array(['age', 'age_group', 'bbox', 'category', 'gender', 'glitch',
       'occlusion', 'out_of_focus', 'photo_quality', 'reid_split', 
       'segmentation_split', 'split', 'turning', 'video'],
-->