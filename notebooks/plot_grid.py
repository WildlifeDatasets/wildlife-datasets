import os
import matplotlib.pyplot as plt
from wildlife_datasets.datasets import *

root = '/data/wildlife_datasets/data'
root_figures = 'figures'
os.makedirs(root_figures, exist_ok=True)
data = [
    (AAUZebraFish, 'auto', False),
    (AerialCattle2017, 'auto', True),
    (AmvrakikosTurtles, 'auto', False),
    (ATRW, 'auto', False),
    (BelugaIDv2, 'auto', True),
    (BirdIndividualIDSegmented, 'auto', True),
    (BristolGorillas2020, 'auto', False),
    (CatIndividualImages, 'auto', False),
    (CTai, 'auto', False),
    (CZoo, 'auto', False),
    (Chicks4FreeID, 'auto', True),
    (CoBRAReIdentificationYoungstock, 'auto', False),
    (CowDataset, 'auto', False),
    (Cows2021v2, 'auto', False),
    (DogFaceNet, 'auto', False),
    (Drosophila, 'auto', False),
    (ELPephants, 'auto', False),
    (FriesianCattle2015v2, 'crop_black', False),
    (FriesianCattle2017, 'auto', False),
    (GiraffeZebraID, 'auto', False),
    (Giraffes, 'auto', False),
    (HappyWhale, 'auto', False),
    (HumpbackWhaleID, 'auto', False),
    (HyenaID2022, 'auto', False),
    (IPanda50, 'auto', False),
    (LeopardID2022, 'auto', False),
    (LionData, 'auto', False),
    (MacaqueFaces, 'auto', False),
    (MPDD, 'auto', False),
    (MultiCamCows2024, 'auto', False),
    (NDD20v2, 'auto', False),
    (NOAARightWhale, 'auto', False),
    (NyalaData, 'auto', False),
    (OpenCows2020, 'auto', True),
    (PolarBearVidID, 'auto', False),
    (PrimFace, 'crop_white', False),
    (ReunionTurtles, 'auto', False),
    (SealIDSegmented, 'crop_black', False),
    (SeaStarReID2023, 'auto', True),
    (SeaTurtleID2022, 'auto', False),
    (SeaTurtleIDHeads, 'auto', False),
    (SMALST, 'auto', False),
    (SouthernProvinceTurtles, 'auto', False),
    (StripeSpotter, 'auto', False),
    (WhaleSharkID, 'auto', False),
    (ZakynthosTurtles, 'auto', False),
    (ZindiTurtleRecall, 'auto', True),
]

for dataset_class, img_load, rotate in data:
    name = dataset_class.display_name()
    dataset = dataset_class(f'{root}/{name}', img_load=img_load)
    dataset.plot_grid(n_rows=2, n_cols=6, rotate=rotate);
    plt.savefig(f'{root_figures}/grid_{name}.png', bbox_inches='tight', dpi=200)