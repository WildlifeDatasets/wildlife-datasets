import os
from typing import Callable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from wildlife_datasets import metrics

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
        matcher: Callable = cosine_similarity,
        k: int = 4,
        return_score: bool = False
        ) -> Tuple[np.ndarray, np.ndarray]:
    """Computes a closest match in the database for each vector in the query set.

    Args:
        features_query (np.ndarray): Query features of size n_query*n_feature. 
        features_database (np.ndarray): Database features of size n_database*n_feature
        ignore (Optional[List[List[int]]], optional): `ignore[i]` is a list of indices
            in the database ignores for i-th query.
        matcher (Callable, optional): function computing similarity.
        k (int, optional): Returned number of predictions.
        return_score (bool, optional): Whether the similalarity is returned.

    Returns:
        Vector of size (n_query,) and array of size (n_query,k). The latter are indices
            in the database for the closest matches (with ignored `ignore` indices).
            If `return_score`, it also returns an array of size (n_query,k) of scores.
    """

    # Create batch chunks
    n_query = len(features_query)
    # If ignore is not provided, initialize as empty
    if ignore is None:
        ignore = [[] for _ in range(n_query)]
    
    idx_true = np.array(range(n_query))
    idx_pred = np.zeros((n_query, k), dtype=np.int32)
    scores = np.zeros((n_query, k))
    # Compute the cosine similarity between the query and the database
    similarity = matcher(features_query, features_database)
    # Set -infinity for ignored indices
    for i in range(len(ignore)):
        similarity[i, ignore[i]] = -np.inf
    # Find the closest matches (k highest values)
    idx_pred = (-similarity).argsort(axis=-1)[:, :k]
    if return_score:
        scores = np.take_along_axis(similarity, idx_pred, axis=-1)
        return idx_true, idx_pred, scores
    else:
        return idx_true, idx_pred

def predict(df, features, col_label='identity', col_split='split'):
    y_pred = np.full(len(df), np.nan, dtype=object)
    similarity_pred = np.full(len(df), np.nan, dtype=object)
    for dataset_name in df['dataset'].unique():
        idx_train = np.where((df['dataset'] == dataset_name) * (df[col_split] == 'train'))[0]
        idx_test = np.where((df['dataset'] == dataset_name) * (df[col_split] == 'test'))[0]

        idx_true, idx_pred, similarity = compute_predictions(features[idx_test], features[idx_train], return_similarity=True)
        idx_true = idx_test[idx_true]
        idx_pred = idx_train[idx_pred]

        y_pred[idx_true] = df[col_label].iloc[idx_pred[:,0]].values
        similarity_pred[idx_true] = similarity[:,0]
    return y_pred, similarity_pred

def compute_baks_baus(df, y_pred, new_individual='', col_label='identity', col_split='split'):
    baks = {}
    baus = {}
    y_true = df[col_label].to_numpy()
    for dataset, df_dataset in df.groupby('dataset'):
        identity_train = df_dataset[col_label][df_dataset[col_split] == 'train'].to_numpy()
        identity_test = df_dataset[col_label][df_dataset[col_split] == 'test'].to_numpy()
        identity_test_only = list(set(identity_test) - set(identity_train))                
        idx = df.index.get_indexer(df_dataset.index[df_dataset[col_split] == 'test'])
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

def make_radar_plot2(df, color, cols, title=None, figsize=(9, 9), fontsize=16, rotation=4.0, line_width=2, s=0.7, file_name=None):
    pi = np.pi
    categories = df.index
    N = len(categories)
    col0 = cols[0]

    # Set the angles for each axis on the radar plot
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    # Initialize the radar chart
    plt.figure(figsize=figsize)
    ax = plt.subplot(1, 1, 1, polar=True)

    # Rotate the radar chart to start at the top
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)

    # Set category labels on the radar chart
    plt.xticks(angles[:-1], categories, color='black', size=fontsize)
    ax.tick_params(axis='x', rotation=rotation)

    # Set y-label positions and limits
    ax.set_rlabel_position(0)
    plt.yticks([])  # Remove default y-ticks
    plt.ylim(0, 1)
    ax.spines['polar'].set_visible(False)
    col_main_values = [s] * N  # Set 'megadescriptor-L-384' values to 90% of the radius
    col_main_values += col_main_values[:1]  # Repeat the first value for closure

    # Plot 'megadescriptor-L-384' as the fixed reference line
    ax.plot(angles, col_main_values, color=color[col0], linewidth=line_width, linestyle='dotted', label=cols[0])
    ax.fill(angles, col_main_values, color=color[col0], alpha=0.1)

    # Display exact values of 'megadescriptor-L-384' near the plot
    for i, angle in enumerate(angles[:-1]):
        value_text = f"{df[col0].values[i]:.1f}"  # Format the number to 2 decimal places
        ax.text(angle, s - 0.1, value_text, horizontalalignment='center', size=fontsize - 3, color='black')

    # Plot relative values for 'avg-local' and 'avg-all'
    for col in cols[1:]:
        # Calculate relative values for each dataset relative to 'megadescriptor-L-384'
        relative_values = (df[col].values / df[col0].values) * s  # Scale relative to 90% of col_main
        relative_values = np.clip(relative_values, 0, 1)  # Ensure values stay within [0, 1]
        relative_values = list(relative_values) + [relative_values[0]]  # Repeat the first value for closure

        # Plot the radar chart for the column
        ax.plot(angles, relative_values, color=color[col], linewidth=line_width, linestyle='solid', label=col)
        ax.fill(angles, relative_values, color=color[col], alpha=0.05)

        # Display exact values near the data points
        for i, angle in enumerate(angles[:-1]):
            value_text = f"{df[col].values[i]:.1f}"  # Format the number to 2 decimal places

            if i in [2, ]:
                ax.text(angle, relative_values[i] - 0.12, value_text, horizontalalignment='center', size=fontsize - 3, color='black')
            else:
                ax.text(angle, relative_values[i] + 0.07, value_text, horizontalalignment='center', size=fontsize - 3, color='black')


    leg = ax.legend(loc='upper right', prop={'size': 12}, bbox_to_anchor=(1.13, 1.05), bbox_transform=ax.transAxes)
    if file_name is not None:
        plt.savefig(file_name)
        
def mean(x, idx=None):
    if idx is None:
        return np.mean(list(x.values()))
    else:
        return np.mean([x[i] for i in idx])

def greedy_similarity_clustering(
        similarity_matrix: np.ndarray,
        similarity_threshold: float
        ) -> List:
    """Performs greedy clustering by adding edges with similarity above a threshold.

    Args:
        similarity_matrix (np.ndarray): 2D array where similarity_matrix[i][j] represents 
                           the similarity between nodes i and j
        similarity_threshold (float): Only add edges with similarity >= this threshold

    Returns:
        List of clusters, each cluster is a list of node indices
    """

    n = len(similarity_matrix)
    
    # Create edge list: (i, j, sim)
    edges = [
        (i, j, similarity_matrix[i][j])
        for i in range(n)
        for j in range(i+1, n)
    ]
    
    # Sort edges by descending similarity
    edges.sort(key=lambda x: x[2], reverse=True)
    
    # Union-Find structure for clustering
    parent = list(range(n))
    
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    
    def union(x, y):
        parent[find(x)] = find(y)
    
    # Add edges if they pass threshold
    for u, v, sim in edges:
        if sim < similarity_threshold:
            break  # Stop if similarity below threshold
        if find(u) != find(v):
            union(u, v)

    # Build clusters
    clusters_dict = {}
    for i in range(n):
        root = find(i)
        clusters_dict.setdefault(root, []).append(i)
    
    clusters = list(clusters_dict.values())
    return clusters

class SplitterByFeatures:
    def __init__(self, path_features, original_splitter, thr, file_name=None):
        self.path_features = path_features
        self.original_splitter = original_splitter
        self.thr = thr
        self.file_name = file_name

    def split(self, df):
        clusters = self.get_clusters(df)
        assert len(df) == len(clusters)
        
        if self.file_name is not None:
            np.save(self.file_name, clusters)

        idx_train0, _ = self.original_splitter.split(df)[0]
        idx_train, idx_test = self.original_splitter.resplit_by_clusters(df, clusters, idx_train0)
        return [(idx_train, idx_test)]
    
    def get_clusters(self, df):
        df = df.reset_index(drop=True)
        features = np.load(self.path_features)
        
        df['cluster'] = np.nan
        for _, df_identity in df.groupby('identity'):
            features_subset = features[df_identity.index]
            sim = cosine_similarity(features_subset, features_subset)
            np.fill_diagonal(sim, -np.inf)

            clusters = greedy_similarity_clustering(sim, self.thr)
            for cluster_id, cluster in enumerate(clusters):
                if len(cluster) > 1:
                    df.loc[df_identity.index[cluster], 'cluster'] = cluster_id
        return df['cluster'].to_numpy()
