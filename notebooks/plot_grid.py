import os

import matplotlib.pyplot as plt

from wildlife_datasets.datasets import *

root = "/data/wildlife_datasets/data"
root_figures = "figures"
os.makedirs(root_figures, exist_ok=True)
data = [
    (AAUZebraFish, "auto", False, None),
    (AerialCattle2017, "auto", True, None),
    (AmvrakikosTurtles, "auto", False, None),
    (ATRW, "auto", False, None),
    (BalearicLizardSegmented, "auto", False, "BalearicLizard"),
    (BelugaIDv2, "auto", True, None),
    (BirdIndividualIDSegmented, "auto", True, None),
    (BristolGorillas2020, "auto", False, None),
    (CatIndividualImages, "auto", False, None),
    (CattleMuzzle, "auto", False, None),
    (CTai, "auto", False, None),
    (CZoo, "auto", False, None),
    (Chicks4FreeID, "auto", True, None),
    (CoBRAReIdentificationYoungstock, "auto", False, None),
    (CowDataset, "auto", False, None),
    (Cows2021v2, "auto", False, None),
    (DogFaceNet, "auto", False, None),
    (Drosophila, "auto", False, None),
    (ELPephants, "auto", False, None),
    (FriesianCattle2015v2, "crop_black", False, None),
    (FriesianCattle2017, "auto", False, None),
    (GiraffeZebraID, "auto", False, None),
    (Giraffes, "auto", False, None),
    (HappyWhale, "auto", False, None),
    (HolsteinCattleRecognition, "auto", False, None),
    (HumpbackWhaleID, "auto", False, None),
    (HyenaID2022, "auto", False, None),
    (IPanda50, "auto", False, None),
    (LeopardID2022, "auto", False, None),
    (LionData, "auto", False, None),
    (MacaqueFaces, "auto", False, None),
    (MPDD, "auto", False, None),
    (MultiCamCows2024, "auto", False, None),
    (NDD20v2, "auto", False, None),
    (NOAARightWhale, "auto", False, None),
    (NyalaData, "auto", False, None),
    (OpenCows2020, "auto", True, None),
    (PolarBearVidID, "auto", False, None),
    (PrimFace, "crop_white", False, None),
    (ReunionTurtles, "auto", False, None),
    (SealIDSegmented, "crop_black", False, None),
    (SeaStarReID2023, "auto", True, None),
    (SeaTurtleID2022, "auto", False, None),
    (SeaTurtleIDHeads, "auto", False, None),
    (SMALST, "auto", False, None),
    (SouthernProvinceTurtles, "auto", False, None),
    (StripeSpotter, "auto", False, None),
    (TurtlesOfSMSRC, "auto", False, None),
    (WildRaptorID, "auto", False, None),
    (WhaleSharkID, "auto", False, None),
    (ZakynthosTurtles, "auto", False, None),
    (ZindiTurtleRecall, "auto", True, None),
]

for dataset_class, img_load, rotate, name in data:
    if name is None:
        name = dataset_class.display_name()
    dataset = dataset_class(f"{root}/{name}", img_load=img_load)
    dataset.plot_grid(n_rows=2, n_cols=6, rotate=rotate)
    plt.savefig(f"{root_figures}/grid_{name}.png", bbox_inches="tight", dpi=200)
