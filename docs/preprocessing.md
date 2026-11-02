# Processing datasets

Some of the datasets require special treatment or the download of extraction works only on Linux. These are described below in the `Processsing requirements` column. We also fixed labels in some datasets as described in the `Dataset modifications` column.

## MetaDatasets

<!-- preprocessing-meta -->


## Datasets

<!-- preprocessing -->

## Manual download and extracting

### Kaggle

Some datasets are stored on Kaggle. To use our automatic download method, follow the [steps](https://www.kaggle.com/docs/api) described in the Installation and Authentication sections.

### AAUZebrafish

[Kaggle requirements](#kaggle) need to be satisfied.

### AmvrakikosTurtles

[Kaggle requirements](#kaggle) need to be satisfied.

### AnimalCLEF2025

[Kaggle requirements](#kaggle) need to be satisfied.

### AnimalCLEF2026

[Kaggle requirements](#kaggle) need to be satisfied.

### BalearicLizard

[Kaggle requirements](#kaggle) need to be satisfied.

### BrownBearHeads

[Kaggle requirements](#kaggle) need to be satisfied.

The dataset is distributed as a prepared Kaggle reupload. The automatic download should work directly:

```python
datasets.BrownBearHeads.get_data('data/BrownBearHeads')
```

The downloaded dataset already contains a prepared `metadata.csv` together with reorganized image paths, so the loader reads the catalogue directly without reconstructing it from the original raw split CSV files.

### BirdIndividualID

The dataset is stored on [Google drive](https://drive.google.com/uc?id=1YT4w8yF44D-y9kdzgF38z2uYbHfpiDOA) but needs to be downloaded manually due to its size. After downloading it, place it into folder ``BirdIndividualID'' and run

```python
datasets.BirdIndividualID.extract('data/BirdIndividualID')
```

to extract it. Do not extract it manually because there is some postprocessing involved.

### BristolGorillas2020

Downloading works only on Linux. Download it manually from its [webpage](https://data.bris.ac.uk/data/dataset/jf0859kboy8k2ufv60dqeb2t8). It may take several retries. Then either extract it manually or call

```python
datasets.BristolGorillas2020.extract('data/BristolGorillas2020')
```

### CatIndividualImages

[Kaggle requirements](#kaggle) need to be satisfied.

### CzechLynx

[Kaggle requirements](#kaggle) need to be satisfied.

### ELPephants

The dataset is stored [privately](https://inf-cv.uni-jena.de/home/research/datasets/elpephants/) and needs to acknowledge license terms before downloading. After doing so, extract it manually or run

```python
datasets.ELPephants.extract('data/ELPephants')
```

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

### MPDD

The dataset is does not have a permanent URL. Download it [manually](https://data.mendeley.com/datasets/v5j6m8dzhv/1). After doing so, extract it manually or run

```python
datasets.MPDD.extract('data/MPDD')
```

### NDD20

The dataset is does not have a permanent URL. Download it [manually](https://doi.org/10.25405/data.ncl.c.4982342). After doing so, extract it manually or run

```python
datasets.NDD20v2.extract('data/NDD20')
```

### NOAARightWhale

[Kaggle requirements](#kaggle) need to be satisfied. Also you need to go to the [competition website](https://www.kaggle.com/c/noaa-right-whale-recognition), the Data tab and agree with terms.

### ReunionTurtles

[Kaggle requirements](#kaggle) need to be satisfied.

### SealID

SealID requires a one-time token for download. Please go their [download website](https://doi.org/10.23729/0f4a3296-3b10-40c8-9ad3-0cf00a5a4a53), click the Data tab, then three dots next to the Download button and copy the `URL` link. Then use

```python
url = '' # Paste the URL here
datasets.SealID.get_data('data/SealID', url=url)
```

### SeaTurtleID2022

[Kaggle requirements](#kaggle) need to be satisfied.

### SMALST

The dataset fails to download automatically. Download it [manually](https://drive.google.com/open?id=1yVy4--M4CNfE5x9wUr1QBmAXEcWb6PWF). After doing so, extract it manually or run

```python
datasets.SMALST.extract('data/SMALST')
```

### SouthernProvinceTurtles

[Kaggle requirements](#kaggle) need to be satisfied.

### StripeSpotter

Extracting works only on Linux. Use

```python
datasets.StripeSpotter.download('data/StripeSpotter')
```

to download the dataset and then extract it manually.

### WildlifeReID10k

[Kaggle requirements](#kaggle) need to be satisfied.

### ZakynthosTurtles

[Kaggle requirements](#kaggle) need to be satisfied.

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
