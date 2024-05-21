import os
import pandas as pd
from wildlife_datasets import datasets

license_conversion = {
    'Missing': 'None',
    'Other': 'Other',
    'Attribution 4.0 International (CC BY 4.0)': 'CC BY 4.0',
    'Creative Commons Attribution 4.0 International': 'CC BY 4.0',
    'Attribution-NonCommercial-ShareAlike 4.0 International': 'CC BY-NC-SA 4.0',
    'Non-Commercial Government Licence for public sector information': 'NC-Government',
    'Community Data License Agreement – Permissive': 'CDLA-Permissive-1.0',
    'Community Data License Agreement – Permissive, Version 1.0': 'CDLA-Permissive-1.0',
    'MIT License': 'MIT',
    'Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)': 'CC BY-NC-SA 4.0',
    'Attribution-ShareAlike 3.0 Unported' : 'CC BY-SA 3.0',
    'Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)': 'CC BY-SA 4.0',
}

def getFolderSize(folder):
    total_size = os.path.getsize(folder)
    for item in os.listdir(folder):
        itempath = os.path.join(folder, item)
        if os.path.isfile(itempath):
            total_size += os.path.getsize(itempath)
        elif os.path.isdir(itempath):
            total_size += getFolderSize(itempath)
    return total_size

def rename_index(df):
    rename = {}
    for dataset_name in df.index:
        try:
            metadata = eval(f'datasets.{dataset_name}.metadata')
            citation = " \cite{" + metadata['cite'] + "}"
        except:
            citation = ''
        rename[dataset_name] = dataset_name + citation
    return df.rename(index=rename)

def load_clusters(identities, save_clusters_prefix):
    clusters_all = {}
    for identity in identities:
        file_name = f'{save_clusters_prefix}_{identity}.csv'
        clusters_all[identity] = []
        if os.path.exists(file_name):
            clusters = pd.read_csv(file_name, index_col=0)
            for cluster_id, cluster in clusters.groupby('cluster'):
                if cluster_id != -1:
                    clusters_all[identity].append(list(cluster.index))
    return clusters_all