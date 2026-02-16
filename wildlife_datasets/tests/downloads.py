from wildlife_datasets.datasets import *

# Datasets which can be downloaded automatically
names_download = [
    (AAUZebraFish, "AAUZebraFish"),
    (AerialCattle2017, "AerialCattle2017"),
    (AmvrakikosTurtles, "AmvrakikosTurtles"),
    (ATRW, "ATRW"),
    (BalearicLizard, "BalearicLizard"),
    (BalearicLizardSegmented, "BalearicLizard"),
    (BelugaIDv2, "BelugaID"),
    (CattleMuzzle, "CattleMuzzle"),
    (CatIndividualImages, "CatIndividualImages"),
    (Chicks4FreeID, None),
    (CoBRAReIdentificationYoungstock, "CoBRAReIdentificationYoungstock"),
    (CowDataset, "CowDataset"),
    (Cows2021v2, "Cows2021"),
    (CzechLynxv2, "CzechLynx"),
    (CTai, "CTai"),
    (CZoo, "CZoo"),
    (DogFaceNet, "DogFaceNet"),
    (FriesianCattle2015v2, "FriesianCattle2015"),
    (FriesianCattle2017, "FriesianCattle2017"),
    (GiraffeZebraID, "GiraffeZebraID"),
    (Giraffes, "Giraffes"),
    (HappyWhale, "HappyWhale"),
    (HolsteinCattleRecognition, "HolsteinCattleRecognition"),
    (HumpbackWhaleID, "HumpbackWhaleID"),
    (HyenaID2022, "HyenaID2022"),
    (IPanda50, "IPanda50"),
    (LeopardID2022, "LeopardID2022"),
    (LionData, "LionData"),
    (MacaqueFaces, "MacaqueFaces"),
    (MultiCamCows2024, "MultiCamCows2024"),
    (NOAARightWhale, "NOAARightWhale"),
    (NyalaData, "NyalaData"),
    (OpenCows2020, "OpenCows2020"),
    (PolarBearVidID, "PolarBearVidID"),
    (PrimFace, "PrimFace"),
    (ReunionTurtles, "ReunionTurtles"),
    (SeaStarReID2023, "SeaStarReID2023"),
    (SeaTurtleID2022, "SeaTurtleID2022"),
    (SeaTurtleIDHeads, "SeaTurtleIDHeads"),
    (SouthernProvinceTurtles, "SouthernProvinceTurtles"),
    (StripeSpotter, "StripeSpotter"),
    (TurtlesOfSMSRC, "TurtlesOfSMSRC"),
    (WildRaptorID, "WildRaptorID"),
    (WhaleSharkID, "WhaleSharkID"),
    (ZakynthosTurtles, "ZakynthosTurtles"),
    (ZindiTurtleRecall, "ZindiTurtleRecall"),
]

# Datasets which have some problems (cannot be downloaded or takes too long)
names_open = [
    (BirdIndividualID, "BirdIndividualID"),
    (BirdIndividualIDSegmented, "BirdIndividualIDSegmented"),
    (BristolGorillas2020, "BristolGorillas2020"),
    (Drosophila, "Drosophila"),
    (ELPephants, "ELPephants"),
    (MPDD, "MPDD"),
    (NDD20v2, "NDD20"),
    (SealID, "SealID"),
    (SealIDSegmented, "SealIDSegmented"),
    (SMALST, "SMALST"),
]

# Set the root folders
root_new = "/data/wildlife_datasets/data_test"
root_old = "/data/wildlife_datasets/data"

# Check whether it downloads correctly
for dataset_class, folder in names_download:
    dataset_class.get_data(f"{root_new}/{folder}")

# Check whether the newly downloaded is the same as before
max_len = max([len(str(folder)) for _, folder in names_download])
for dataset_class, folder in names_download:
    dataset_new = dataset_class(f"{root_new}/{folder}")
    dataset_old = dataset_class(f"{root_old}/{folder}")
    is_same = dataset_new.metadata.equals(dataset_old.metadata)
    name = folder if folder is not None else dataset_new.display_name()
    print(f"{name:<{max_len}}: {is_same}")

# Check whether the other datasets can load
max_len = max([len(str(folder)) for _, folder in names_open])
for dataset_class, folder in names_open:
    dataset_class(f"{root_old}/{folder}")
    print(f"{folder:<{max_len}} loaded succesfully")