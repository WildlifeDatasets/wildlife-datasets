import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, List, Tuple
from wildlife_datasets import datasets, metrics
from wildlife_tools.similarity import CosineSimilarity

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
            summary = eval(f'datasets.{dataset_name}.summary')
            citation = " \cite{" + summary['cite'] + "}"
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

def compute_predictions(
        features_query: np.ndarray,
        features_database: np.ndarray,
        ignore: Optional[List[List[int]]] = None,
        k: int = 4,
        batch_size: int = 1000,
        return_similarity: bool = False,
        ) -> Tuple[np.ndarray, np.ndarray]:
    """Computes a closest match in the database for each vector in the query set.

    Args:
        features_query (np.ndarray): Query features of size n_query*n_feature. 
        features_database (np.ndarray): Database features of size n_database*n_feature
        ignore (Optional[List[List[int]]], optional): `ignore[i]` is a list of indices
            in the database ignores for i-th query.
        k (int, optional): Returned number of predictions.
        batch_size (int, optional): Size of the computation batch.
        return_similarity (bool, optional): Whether similarity scores are returned.

    Returns:
        Vector of size (n_query,) and array of size (n_query,k). The latter are indices
            in the database for the closest matches (with ignored `ignore` indices)
    """

    # Create batch chunks
    n_query = len(features_query)
    n_chunks = int(np.ceil(n_query / batch_size))
    chunks = np.array_split(range(n_query), n_chunks)
    # If ignore is not provided, initialize as empty
    if ignore is None:
        ignore = [[] for _ in range(n_query)]
    
    matcher = CosineSimilarity()
    idx_true = np.array(range(n_query))
    idx_pred = np.zeros((n_query, k), dtype=np.int32)
    similarity_pred = np.zeros((n_query, k))
    for chunk in chunks:
        # Compute the cosine similarity between the query chunk and the database
        similarity = matcher(query=features_query[chunk], database=features_database)['cosine']
        # Set -infinity for ignored indices
        for i in range(len(chunk)):
            similarity[i, ignore[chunk[i]]] = -np.inf
        # Find the closest matches (k highest values)
        idx_pred[chunk,:] = (-similarity).argsort(axis=-1)[:, :k]
        if return_similarity:
            for i in range(len(chunk)):
                similarity_pred[chunk[i],:] = similarity[i, idx_pred[chunk[i],:]]
    if return_similarity:        
        return idx_true, idx_pred, similarity_pred
    else:
        return idx_true, idx_pred

def predict(df, features, split_col='split'):
    y_pred = np.full(len(df), np.nan, dtype=object)
    similarity_pred = np.full(len(df), np.nan, dtype=object)
    for dataset_name in df['dataset'].unique():
        idx_train = np.where((df['dataset'] == dataset_name) * (df[split_col] == 'train'))[0]
        idx_test = np.where((df['dataset'] == dataset_name) * (df[split_col] == 'test'))[0]

        idx_true, idx_pred, similarity = compute_predictions(features[idx_test], features[idx_train], return_similarity=True)
        idx_true = idx_test[idx_true]
        idx_pred = idx_train[idx_pred]

        y_pred[idx_true] = df['identity'].iloc[idx_pred[:,0]].values
        similarity_pred[idx_true] = similarity[:,0]
    return y_pred, similarity_pred

def compute_baks_baus(df, y_pred, new_individual='', split_col='split'):
    baks = {}
    baus = {}
    y_true = df['identity'].to_numpy()
    for dataset, df_dataset in df.groupby('dataset'):
        identity_train = df_dataset['identity'][df_dataset[split_col] == 'train'].to_numpy()
        identity_test = df_dataset['identity'][df_dataset[split_col] == 'test'].to_numpy()
        identity_test_only = list(set(identity_test) - set(identity_train))                
        idx = df.index.get_indexer(df_dataset.index[df_dataset[split_col] == 'test'])
        baks[dataset] = metrics.BAKS(y_true[idx], y_pred[idx], identity_test_only)
        baus[dataset] = metrics.BAUS(y_true[idx], y_pred[idx], identity_test_only, new_individual)
    return baks, baus

def make_radar_plot(df, color, title=None, use_col='metric', figsize=(9, 9), fontsize=22, rotation=4.0):
    pi = np.pi
    categories=df.index
    N = len(categories)
    
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    plt.rc('figure', figsize=figsize)
 
    ax = plt.subplot(1,1,1, polar=True)
 
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
 
    plt.xticks(angles[:-1], categories, color='black', size=fontsize)
    ax.tick_params(axis='x', rotation=rotation)
 
    ax.set_rlabel_position(0)
    plt.yticks([0.20,0.40,0.60,0.80], ["0.2","0.4","0.6","0.8"], color="black", size=fontsize)
    plt.ylim(0,1)
    
    values = list(df[use_col].values)
    values += values[:1]
    ax.plot(angles, values, color = color, linewidth=1, linestyle='solid', label=use_col)
    ax.fill(angles, values, color = color, alpha = 0.15)
    
    if title:
        plt.title(title, fontsize=20, x = 0.5, y = 1.1)