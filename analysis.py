import os
import numpy as np
from datasets import *

info_datasets = [
    (AAUZebraFishID, {}),
    (AerialCattle2017, {}),
    (ATRW, {}),
    (BelugaID, {}),
    (BirdIndividualID, {'variant': 'source'}),
    (BirdIndividualID, {'variant': 'segmented'}),
    (CTai, {}),
    (CZoo, {}),
    (Cows2021, {}),
    (Drosophila, {}),
    (FriesianCattle2015, {}),
    (FriesianCattle2017, {}),
    (GiraffeZebraID, {}),
    (Giraffes, {}),
    (HappyWhale, {}),
    (HumpbackWhaleID, {}),
    (HyenaID2022, {}),
    (IPanda50, {}),
    (LeopardID2022, {}),
    (LionData, {}),
    (MacaqueFaces, {}),
    (NDD20, {}),
    (NOAARightWhale, {}),
    (NyalaData, {}),
    (OpenCows2020, {}),
    (SealID, {'variant': 'source'}),
    (SealID, {'variant': 'segmented'}),
    (SMALST, {}),
    (StripeSpotter, {}),
    (WhaleSharkID, {}),
    (WNIGiraffes, {}),
    (ZindiTurtleRecall, {})
]

def unique_datasets_list(datasets_list):
    _, idx = np.unique([dataset[0].__name__ for dataset in datasets_list], return_index=True)
    idx = np.sort(idx)

    datasets_list_red = []
    for i in idx:
        datasets_list_red.append(datasets_list[i])

    return datasets_list_red

def add_paths(datasets_list, root_dataset, root_dataframe):
    datasets_list_mod = []
    for dataset in datasets_list:        
        if len(dataset[1]) == 0:                        
            csv_path = os.path.join(root_dataframe, dataset[0].__name__ + '.csv')
        else:
            csv_path = os.path.join(root_dataframe, dataset[0].__name__ + '_' + dataset[1]['variant'] + '.csv')
        datasets_list_mod.append(dataset + (os.path.join(root_dataset, dataset[0].__name__),) + (csv_path,))
    return datasets_list_mod

