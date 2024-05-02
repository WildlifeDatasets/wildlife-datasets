# Processing datasets

Some of the datasets require special treatment or the download of extraction works only on Linux. These are described below in the `Processsing requirements` column. We also fixed labels in some datasets as described in the `Dataset modifications` column.

| Dataset                |              Processing requirements           | Dataset modifications |
|------------------------|:----------------------|:---------|
| [AAUZebrafish](https://www.kaggle.com/datasets/aalborguniversity/aau-zebrafish-reid)          | Kaggle required                   |                  |
| [AerialCattle2017](https://data.bris.ac.uk/data/dataset/3owflku95bxsx24643cybxu3qh)          |                                |                  |
| [ATRW](https://lila.science/datasets/atrw)                   |                                |                  |
| [BelugaID](https://lila.science/datasets/beluga-id-2022/)              |                                |                  |
| [BirdIndividualID](https://github.com/AndreCFerreira/Bird_individualID)     | Manual download    | Few images removed |
| [CatIndividualImages](https://www.kaggle.com/datasets/timost1234/cat-individuals)          | Kaggle required                   |                  |
| [CTai](https://github.com/cvjena/chimpanzee_faces)       |                                | Labels fixed |
| [CZoo](https://github.com/cvjena/chimpanzee_faces)       |                                |                  |
| [CowDataset](https://figshare.com/articles/dataset/data_set_zip/16879780)                   |                                |                  |
| [Cows2021](https://data.bris.ac.uk/data/dataset/4vnrca7qw1642qlwxjadp87h7)              |                                | Labels fixed  |
| [DogFaceNet](https://github.com/GuillaumeMougeot/DogFaceNet)          |                    |                  |
| [Drosophila](https://github.com/j-schneider/fly_eye)             |                                | Few images removed |
| [FriesianCattle2015](https://data.bris.ac.uk/data/dataset/wurzq71kfm561ljahbwjhx9n3)   |                                | Labels fixed |
| [FriesianCattle2017](https://data.bris.ac.uk/data/dataset/2yizcfbkuv4352pzc32n54371r)   |                                |                  |
| [Giraffes](ftp://pbil.univ-lyon1.fr/pub/datasets/miele2021)       | Only linux: downloading                               |                  |
| [GiraffeZebraID](https://lila.science/datasets/great-zebra-giraffe-id)       |                                |                  |
| [HappyWhale](https://www.kaggle.com/competitions/happy-whale-and-dolphin)             | Kaggle required + terms   | Species fixed |
| [HumpbackWhaleID](https://www.kaggle.com/competitions/humpback-whale-identification)          | Kaggle required + terms   | Unknown animals renamed |
| [HyenaID2022](https://lila.science/datasets/hyena-id-2022/)               |                                |                  |
| [IPanda50](https://github.com/iPandaDateset/iPanda-50)              |                                | Few image renamed |
| [LeopardID2022](https://lila.science/datasets/leopard-id-2022/)             |                                | Unknown animals renamed |
| [LionData](https://github.com/tvanzyl/wildlife_reidentification)              |                                |                  |
| [MacaqueFaces](https://github.com/clwitham/MacaqueFaces)          |                                |                  |
| [MPDD](https://data.mendeley.com/datasets/v5j6m8dzhv/1)          |                                |                  |
| [NDD20](https://doi.org/10.25405/data.ncl.c.4982342)                  |                                |                  |
| [NOAARightWhale](https://www.kaggle.com/c/noaa-right-whale-recognition)        | Kaggle required + terms   |                  |
| [NyalaData](https://github.com/tvanzyl/wildlife_reidentification)             |                                |                  |
| [OpenCows2020](https://data.bris.ac.uk/data/dataset/10m32xl88x2b61zlkkgz3fml17)           |                                |                  |
| [PolarBearVidID](https://zenodo.org/records/7564529)          |                                |                  |
| [SealID](https://doi.org/10.23729/0f4a3296-3b10-40c8-9ad3-0cf00a5a4a53)                 | Download with single use token |                  |
| [SeaStarReID2023](https://lila.science/sea-star-re-id-2023/)          |                                |                  |
| [SeaTurtleID](https://www.kaggle.com/datasets/wildlifedatasets/seaturtleid)                 | Kaggle required                   |                  |
| [SMALST](https://github.com/silviazuffi/smalst)                 | Only linux: extracting            |                  |
| [StripeSpotter](https://code.google.com/archive/p/stripespotter/downloads)          | Only linux: extracting            |                  |
| [WhaleSharkID](https://lila.science/datasets/whale-shark-id)         |                                |                  |
| [ZindiTurtleRecall](https://zindi.africa/competitions/turtle-recall-conservation-challenge)    |                                |                  |

## Manual download and extracting

### Kaggle

Some datasets are stored on Kaggle. To use our automatic download method, follow the [steps](https://www.kaggle.com/docs/api) described in the Installation and Authentication sections.

### AAUZebrafish

[Kaggle requirements](#kaggle) need to be satisfied.

### BirdIndividualID

The dataset is stored on [Google drive](https://drive.google.com/uc?id=1YT4w8yF44D-y9kdzgF38z2uYbHfpiDOA) but needs to be downloaded manually due to its size. After downloading it, place it into folder ``BirdIndividualID'' and run

```python
datasets.BirdIndividualID.extract('data/BirdIndividualID')
```

to extract it. Do not extract it manually because there is some postprocessing involved.

### CatIndividualImages

[Kaggle requirements](#kaggle) need to be satisfied.

### Giraffes

Downloading works only on Linux. Download it manually from the [FTP server](ftp://pbil.univ-lyon1.fr/pub/datasets/miele2021/) by using some client such as [FileZilla](https://filezilla-project.org/download.php?type=client).

### HappyWhale

[Kaggle requirements](#kaggle) need to be satisfied. Also you need to go to the [competition website](https://www.kaggle.com/competitions/happy-whale-and-dolphin), the Data tab and agree with terms.

### HumpbackWhale

[Kaggle requirements](#kaggle) need to be satisfied. Also you need to go to the [competition website](https://www.kaggle.com/competitions/humpback-whale-identification), the Data tab and agree with terms.

### IPanda50

IPanda50 may fail to download files because of Google Drive quotas. If this happens, download three zip files manually as described in this [Github repository](https://github.com/iPandaDateset/iPanda-50). Then either extract them manually or run

```python
datasets.IPanda50.extract('data/IPanda50')
```

### NOAARightWhale

[Kaggle requirements](#kaggle) need to be satisfied. Also you need to go to the [competition website](https://www.kaggle.com/c/noaa-right-whale-recognition), the Data tab and agree with terms.

### SealID

SealID requires a one-time token for download. Please go their [download website](https://doi.org/10.23729/0f4a3296-3b10-40c8-9ad3-0cf00a5a4a53), click the Data tab, then three dots next to the Download button and copy the `URL` link. Then use

```python
url = '' # Paste the URL here
datasets.SealID.get_data('data/SealID', url=url)
```

### SeaTurtleID

[Kaggle requirements](#kaggle) need to be satisfied.

### SMALST

Extracting works only on Linux. Use

```python
datasets.SMALST.download('data/SMALST')
```

to download the dataset and then extract it manually.

### StripeSpotter

Extracting works only on Linux. Use

```python
datasets.StripeSpotter.download('data/StripeSpotter')
```

to download the dataset and then extract it manually.

## Dataset modifications

### BirdIndividualID

We removed images containing multiple birds, where it not indicated which bird is which. We also removed a few cropped images due to different folder structure.

### CTai

There were several misspelled labels, which we corrected. Mainly the correct forms read Akrouba (instead of Akouba), Fredy (Freddy), Ibrahim (Ibrahiim), Lilou (Liliou), Wapi (Wapii) and Woodstock (Woodstiock). Also the identity Adult was not a proper identity and we replaced it with `self.unknown_name`.

### Cows2021

We ignored all images not in the `Identification` folder because they were focused on detection. We also ignored all images in the `Identification/Train` because some folders contained images of different individuals.

We merged identity 105 into 29 and 164 into 148 as they depicted the same individuals. We performed the following correction:

| Image name | Old identity | New identity |
|:-----|:-----|:-----|
| image_0001226_2020-02-11_12-43-7_roi_001.jpg | 137 | 107 | 

### Drosophila

We removed a few images due to different folder structure.

### FriesianCattle2015

Images in the folder `Cows-training` were removed as they are identical to images in the folder `Cows-testing`. 

Multiple individuals were removed as the images are copies of images of different individuals. Namely we removed 19 (duplicate of 15), 20 (18), 21 (17), 22 (16), 23 (11), 24 (14), 25 (13), 26 (12), 27 (11), 28 (13), 29 (12), 31 (17), 32 (16), 33 (18) and 37 (30).

### HappyWhale

We fixed the typos in species bottlenose_dolpin and kiler_whale.

### HumpbackWhaleID

We replaced the `new_whale` by `self.unknown_name`.

### IPanda50

We renamed imaged containing non-ASCII characters.

### LeopardID2022

We replaced the `____` by `self.unknown_name`.
