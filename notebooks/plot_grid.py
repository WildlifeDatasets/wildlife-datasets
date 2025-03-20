import os
import matplotlib.pyplot as plt
from wildlife_datasets.datasets import *

root = '/data/wildlife_datasets/data'
root_figures = 'figures'
os.makedirs(root_figures, exist_ok=True)
data = [
    (AAUZebraFish, 'auto'),
    (AerialCattle2017, 'auto'),
    (AmvrakikosTurtles, 'auto'),
    (ATRW, 'auto'),
    (BelugaIDv2, 'auto'),
    (BirdIndividualIDSegmented, 'auto'),
    (CatIndividualImages, 'auto'),
    (CTai, 'auto'),
    (CZoo, 'auto'),
    (Chicks4FreeID, 'auto'),
    (CowDataset, 'auto'),
    (Cows2021v2, 'auto'),
    (DogFaceNet, 'auto'),
    (Drosophila, 'auto'),
    (ELPephants, 'auto'),
    (FriesianCattle2015v2, 'crop_black'),
    (FriesianCattle2017, 'auto'),
    (GiraffeZebraID, 'auto'),
    (Giraffes, 'auto'),
    (HappyWhale, 'auto'),
    (HumpbackWhaleID, 'auto'),
    (HyenaID2022, 'auto'),
    (IPanda50, 'auto'),
    (LeopardID2022, 'auto'),
    (LionData, 'auto'),
    (MacaqueFaces, 'auto'),
    (MPDD, 'auto'),
    (MultiCamCows2024, 'auto'),
    (NDD20v2, 'auto'),
    (NOAARightWhale, 'auto'),
    (NyalaData, 'auto'),
    (OpenCows2020, 'auto'),
    (PolarBearVidID, 'auto'),
    (PrimFace, 'crop_white'),
    (ReunionTurtles, 'auto'),
    (SealIDSegmented, 'crop_black'),
    (SeaStarReID2023, 'auto'),
    (SeaTurtleID2022, 'auto'),
    (SeaTurtleIDHeads, 'auto'),
    (SMALST, 'auto'),
    (SouthernProvinceTurtles, 'auto'),
    (StripeSpotter, 'auto'),
    (WhaleSharkID, 'auto'),
    (ZakynthosTurtles, 'auto'),
    (ZindiTurtleRecall, 'auto'),
]

for dataset_class, img_load in data:
    name = dataset_class.display_name()
    dataset = dataset_class(f'{root}/{name}', img_load=img_load)
    dataset.plot_grid(n_rows=4, n_cols=6, rotate=True);
    plt.savefig(f'{root_figures}/grid_{name}.png', bbox_inches='tight', dpi=200)