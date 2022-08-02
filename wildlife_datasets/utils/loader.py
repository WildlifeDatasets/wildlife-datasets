import os
import numpy as np
import pandas as pd
from .. import datasets

info_datasets_full = [
    (datasets.AAUZebraFishID, {}),
    (datasets.AerialCattle2017, {}),
    (datasets.ATRW, {}),
    (datasets.BelugaID, {}),
    (datasets.BirdIndividualID, {'variant': 'source'}),
    (datasets.BirdIndividualID, {'variant': 'segmented'}),
    (datasets.CTai, {}),
    (datasets.CZoo, {}),
    (datasets.Cows2021, {}),
    (datasets.Drosophila, {}),
    (datasets.FriesianCattle2015, {}),
    (datasets.FriesianCattle2017, {}),
    (datasets.GiraffeZebraID, {}),
    (datasets.Giraffes, {}),
    (datasets.HappyWhale, {}),
    (datasets.HumpbackWhaleID, {}),
    (datasets.HyenaID2022, {}),
    (datasets.IPanda50, {}),
    (datasets.LeopardID2022, {}),
    (datasets.LionData, {}),
    (datasets.MacaqueFaces, {}),
    (datasets.NDD20, {}),
    (datasets.NOAARightWhale, {}),
    (datasets.NyalaData, {}),
    (datasets.OpenCows2020, {}),
    (datasets.SealID, {'variant': 'source'}),
    (datasets.SealID, {'variant': 'segmented'}),
    (datasets.SMALST, {}),
    (datasets.StripeSpotter, {}),
    (datasets.WhaleSharkID, {}),
    (datasets.WNIGiraffes, {}),
    (datasets.ZindiTurtleRecall, {})
]

def unique_datasets_list(datasets_list):
    _, idx = np.unique([dataset[0].__name__ for dataset in datasets_list], return_index=True)
    idx = np.sort(idx)

    datasets_list_red = []
    for i in idx:
        datasets_list_red.append(datasets_list[i])

    return datasets_list_red

def load_datasets(info_datasets, root_dataset, root_dataframe):
    # TODO: some overwrite would be nice
    if not os.path.exists(root_dataframe):
        os.makedirs(root_dataframe)
    datasets = []
    for info_dataset in info_datasets:
        root = os.path.join(root_dataset, info_dataset[0].__name__)
        if len(info_dataset[1]) == 0:
            df_path = os.path.join(root_dataframe, info_dataset[0].__name__ + '.pkl')
        else:
            df_path = os.path.join(root_dataframe, info_dataset[0].__name__ + '_' + info_dataset[1]['variant'] + '.pkl')
        
        if os.path.exists(root) and os.path.exists(df_path):
            df = pd.read_pickle(df_path)
            dataset = info_dataset[0](root, df, download=False, **info_dataset[1])
        elif os.path.exists(root) and not os.path.exists(df_path):
            dataset = info_dataset[0](root, None, download=False, **info_dataset[1])
            dataset.df.to_pickle(df_path)
        elif not os.path.exists(root) and os.path.exists(df_path):
            raise(Exception('Data not found but dataframe found. This should not happen.'))
        elif not os.path.exists(root) and not os.path.exists(df_path):
            dataset = info_dataset[0](root, None, download=True, **info_dataset[1])
            dataset.df.to_pickle(df_path)
        datasets.append(dataset)

    return datasets

