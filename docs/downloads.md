# Troubleshooting: downloads

Some of the datasets require special treatment or the download of extraction works only on Linux. These are described below in the `Requirements` column.

| Dataset                | Method  |             Requirements           |
|------------------------|:-------|:-------------------------------|
| [AAUZebrafish](https://www.kaggle.com/datasets/aalborguniversity/aau-zebrafish-reid)          | AUTO    | Kaggle required                   |
| [AerialCattle2017](https://data.bris.ac.uk/data/dataset/3owflku95bxsx24643cybxu3qh)          | AUTO    |                                |
| [ATRW](https://lila.science/datasets/atrw)                   | AUTO    |                                |
| [BelugaID](https://lila.science/datasets/beluga-id-2022/)              | AUTO    |                                |
| [BirdIndividualID](https://github.com/AndreCFerreira/Bird_individualID)     | MANUAL  | Manual download    |
| [C-Tai](https://github.com/cvjena/chimpanzee_faces)       | AUTO    |                                |
| [C-Zoo](https://github.com/cvjena/chimpanzee_faces)       | AUTO    |                                |
| [Cows2021](https://data.bris.ac.uk/data/dataset/4vnrca7qw1642qlwxjadp87h7)              | AUTO    |                                |
| [Drosophila](https://github.com/j-schneider/fly_eye)             | AUTO    |                                |
| [FriesianCattle2015](https://data.bris.ac.uk/data/dataset/wurzq71kfm561ljahbwjhx9n3)   | AUTO    |                                |
| [FriesianCattle2017](https://data.bris.ac.uk/data/dataset/2yizcfbkuv4352pzc32n54371r)   | AUTO    |                                |
| [Giraffes](ftp://pbil.univ-lyon1.fr/pub/datasets/miele2021)       | AUTO    | Only linux: downloading                               |
| [GiraffeZebraID](https://lila.science/datasets/great-zebra-giraffe-id)       | AUTO    |                                |
| [HappyWhale](https://www.kaggle.com/competitions/happy-whale-and-dolphin)             | MANUAL  | Kaggle required + terms   |
| [HumpbackWhale](https://www.kaggle.com/competitions/humpback-whale-identification)          | MANUAL  | Kaggle required + terms   |
| [HyenaID](https://lila.science/datasets/hyena-id-2022/)               | AUTO    |                                |
| [iPanda-50](https://github.com/iPandaDateset/iPanda-50)              | AUTO    |                                |
| [LeopardID](https://lila.science/datasets/leopard-id-2022/)             | AUTO    |                                |
| [LionData](https://github.com/tvanzyl/wildlife_reidentification)              | AUTO    |                                |
| [MacaqueFaces](https://github.com/clwitham/MacaqueFaces)          | AUTO    |                                |
| [NDD20](https://doi.org/10.25405/data.ncl.c.4982342)                  | AUTO    |                                |
| [NOAARightWhale](https://www.kaggle.com/c/noaa-right-whale-recognition)        | MANUAL  | Kaggle required + terms   |
| [NyalaData](https://github.com/tvanzyl/wildlife_reidentification)             | AUTO    |                                |
| [OpenCows2020](https://data.bris.ac.uk/data/dataset/10m32xl88x2b61zlkkgz3fml17)           | AUTO    |                                |
| [SealID](https://doi.org/10.23729/0f4a3296-3b10-40c8-9ad3-0cf00a5a4a53)                 | MANUAL  | Download with single use token |
| [SeaTurtleID](https://www.kaggle.com/datasets/wildlifedatasets/seaturtleid)                 | AUTO    | Kaggle required                   |
| [SMALST](https://github.com/silviazuffi/smalst)                 | AUTO    | Only linux: extracting            |
| [StripeSpotter](https://code.google.com/archive/p/stripespotter/downloads)          | AUTO    | Only linux: extracting            |
| [WhaleSharkID](https://lila.science/datasets/whale-shark-id)         | AUTO    |                                |
| [WNIGiraffes](https://lila.science/datasets/wni-giraffes)           | AUTO    | Only linux: downloading          |
| [ZindiTurtleRecall](https://zindi.africa/competitions/turtle-recall-conservation-challenge)    | AUTO    |                                |

## Kaggle CLI

Some datasets are stored on Kaggle. To use our automatic download method, follow the [steps](https://www.kaggle.com/docs/api) described in the Installation and Authentication sections.

## AAUZebrafish

[Kaggle requirements](#kaggle) need to be satisfied.

## BirdIndividualID

The dataset is stored on [Google drive](https://drive.google.com/uc?id=1YT4w8yF44D-y9kdzgF38z2uYbHfpiDOA) but needs to be downloaded manually due to its size. After downloading it, place it into folder ``BirdIndividualID'' and run

```python
datasets.SealID.extract('data/BirdIndividualID')
```

to extract it. Do not extract it manually because there is some postprocessing involved.

## Giraffes

Downloading works only on Linux. Download it manually from the [FTP server](ftp://pbil.univ-lyon1.fr/pub/datasets/miele2021/) by using some client such as [FileZilla](https://filezilla-project.org/download.php?type=client).

## HappyWhale

[Kaggle requirements](#kaggle) need to be satisfied. Also you need to go to the [competition website](https://www.kaggle.com/competitions/happy-whale-and-dolphin), the Data tab and agree with terms.

## HumpbackWhale

[Kaggle requirements](#kaggle) need to be satisfied. Also you need to go to the [competition website](https://www.kaggle.com/competitions/humpback-whale-identification), the Data tab and agree with terms.

## NOAARightWhale

[Kaggle requirements](#kaggle) need to be satisfied. Also you need to go to the [competition website](https://www.kaggle.com/c/noaa-right-whale-recognition), the Data tab and agree with terms.

## SealID

SealID requires a one-time token for download. Please go their [download website](https://doi.org/10.23729/0f4a3296-3b10-40c8-9ad3-0cf00a5a4a53), click the Data tab, then three dots next to the Download button and copy the `URL` link. Then use

```python
url = '' # Paste the URL here
datasets.SealID.get_data('data/SealID', url=url)
```

## SeaTurtleID

[Kaggle requirements](#kaggle) need to be satisfied.

## SMALST

Extracting works only on Linux. Use

```python
datasets.SMALST.download('data/SMALST')
```

to download the dataset and then extract it manually.

## StripeSpotter

Extracting works only on Linux. Use

```python
datasets.StripeSpotter.download('data/StripeSpotter')
```

to download the dataset and then extract it manually.

## WNIGiraffes

Even though it is possible to download WNIGiraffes automatically, due to its size, it is highly recommended to use AzCopy. Go to the [download side](https://lila.science/datasets/wni-giraffes) and download the two files using [AzCopy](https://lila.science/faq#downloadtips). Then either manually extract them or run

```python
datasets.WNIGiraffes.extract('data/WNIGiraffes')
```
