import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from tqdm import tqdm
from typing import List, Tuple, Optional
from .lcg import Lcg


class BalancedSplit():
    """Base class for splitting datasets into training and testing sets.

    Implements methods from [this paper](https://arxiv.org/abs/2211.10307).
    Its subclasses need to implement the `split` method.
    It should perform balanced splits separately for all classes.
    Its children are `IdentitySplit` and `TimeAwareSplit`.
    `IdentitySplit` has children `ClosedSetSplit`, `OpenSetSplit` and `DisjointSetSplit`.
    `TimeAwareSplit` has children `TimeProportionSplit` and `TimeCutoffSplit`.
    """

    def __init__(
            self,
            seed: int = 666,
            identity_skip: str = 'unknown',
            col_label: str = 'identity',
            disable_tqdm: bool = True,            
            ) -> None:
        
        self.seed = seed
        self.identity_skip = identity_skip
        self.col_label = col_label
        self.disable_tqdm = disable_tqdm

    def modify_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepares dataframe for splits.

        Removes identities specified in `self.identity_skip` (usually unknown identities).

        Args:
            df (pd.DataFrame): A dataframe of the data. It must contain columns `identity` and `date`.

        Returns:
            Modified dataframe of the data.
        """
        
        df = df.copy()
        df = df[df[self.col_label] != self.identity_skip]
        return df
    
    def initialize_lcg(self) -> Lcg:
        """Returns the random number generator.

        Returns:
            The random number generator.
        """
        
        return Lcg(self.seed)
    
    def split(self, *args, **kwargs) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Splitting method which needs to be implemented by subclasses.

        It splits the dataframe `df` into labels `idx_train` and `idx_test`.
        The subdataset is obtained by `df.loc[idx_train]` (not `iloc`).

        Returns:
            List of splits. Each split is list of labels of the training and testing sets.
        """

        raise(NotImplementedError('Subclasses should implement this. \n You may want to use ClosedSetSplit instead of BalancedSplit.'))

    def resplit_random(
            self,
            df: pd.DataFrame,
            idx_train: np.ndarray,
            idx_test: np.ndarray
            ) -> Tuple[np.ndarray, np.ndarray]:
        """Creates a random re-split of an already existing split.

        The re-split mimics the split as the training set contains
        the same number of samples for EACH individual.
        The same goes for the testing set.

        Args:
            df (pd.DataFrame): A dataframe of the data. It must contain columns `identity` and `date`.
            idx_train (np.ndarray): Labels of the training set.
            idx_test (np.ndarray): Labels of the testing set.

        Returns:
            List of labels of the training and testing sets.
        """

        df = self.modify_df(df)

        # Initialize the random number generator
        lcg = self.initialize_lcg()
        
        # Compute the number of samples for each individual in the training set
        counts_train = {}
        for name, df_name in df.loc[idx_train].groupby(self.col_label):
            counts_train[name] = len(df_name)
        # Compute the number of samples for each individual in the testing set
        counts_test = {}
        for name, df_name in df.loc[idx_test].groupby(self.col_label):
            counts_test[name] = len(df_name)

        idx_train_new = []
        idx_test_new = []
        # Loop over all individuals
        for name, df_name in df.groupby(self.col_label):
            # Extract the number of individuals in the training and testing sets
            n_train = counts_train.get(name, 0)
            n_test = counts_test.get(name, 0)
            if n_train+n_test > 0:
                if len(df_name) < n_train+n_test:
                    raise(Exception('The set is too small.'))
                # Get the correct number of indices in both sets
                idx_permutation = lcg.random_permutation(n_train+n_test)
                idx_permutation = np.array(idx_permutation)
                idx_train_new += list(df_name.index[idx_permutation[:n_train]])
                idx_test_new += list(df_name.index[idx_permutation[n_train:n_train+n_test]])
        return np.array(idx_train_new), np.array(idx_test_new)

    def resplit_by_features(
            self,
            df: pd.DataFrame,
            features: np.ndarray,
            idx_train: np.ndarray,
            save_clusters_prefix: Optional[str] = None,
            **kwargs,
            ) -> Tuple[np.ndarray, np.ndarray]:
        """Creates a random re-split of an already existing split.

        The re-split is based on similarity of features.
        It runs DBSCAN as described in `compute_clusters` and
        performs the clustering as described in `resplit_by_clusters`.
        
        Args:
            df (pd.DataFrame): A dataframe of the data. It must contain column `identity`.
            features (np.ndarray): An array of features with the same length as `df`.
            idx_train (np.ndarray): Labels of the training set.
            save_clusters_prefix (Optional[bool], optional): File name prefix for saving clusters.
            **kwargs (type, optional): See kwargs in `compute_clusters`.

        Returns:
            List of labels of the training and testing sets.
        """        
        
        clusters = self.compute_clusters(df, features, **kwargs)
        if save_clusters_prefix is not None:
            np.save(f'{save_clusters_prefix}.npy', clusters)
        return self.resplit_by_clusters(df, clusters, idx_train)

    def compute_clusters(
            self,
            df: pd.DataFrame,
            features: np.ndarray,
            n_max_cluster: int = 5,
            eps_min: float = 0.01,
            eps_max: float = 0.50,
            eps_step: float = 0.01,
            min_samples: int = 2,
            ) -> np.ndarray:

        """Computes clusters for a random re-split of an already existing split.

        It runs DBSCAN with increasing eps (cluster radius) until
        the clusters are smaller than `n_max_cluster`.
        
        Args:
            df (pd.DataFrame): A dataframe of the data. It must contain column `identity`.
            features (np.ndarray): An array of features with the same length as `df`.
            n_max_cluster (int, optional): Maximal size of cluster before `eps` stops increasing.
            eps_min (float, optional): Lower bound for epsilon.
            eps_max (float, optional): Upper bound for epsilon.
            eps_step (float, optional): Step for epsilon.
            min_samples (int, optional): Minimal cluster size.

        Returns:
            List of clusters.
        """

        df = self.modify_df(df)
        df['cluster'] = np.nan

        for _, df_identity in tqdm(df.groupby(self.col_label), disable=self.disable_tqdm):
            f = features[df.index.get_indexer(df_identity.index)]
            # Run DBScan with increasing eps until there are no clusters bigger than n_max_cluster 
            clusters_saved = None
            for eps in np.arange(eps_min, eps_max+eps_step, eps_step):
                clustering = DBSCAN(eps=eps, min_samples=min_samples)
                clustering.fit(f)
                clusters = pd.Series(clustering.labels_)
                clusters_counts = clusters.value_counts(sort=True)
                # Check if the largest clusters (without outliers) is not too big
                if clusters_counts.index[0] == -1:
                    clustering_failed = len(clusters_counts) > 1 and clusters_counts.iloc[1] > n_max_cluster
                else:
                    clustering_failed = len(clusters_counts) == 1 or clusters_counts.iloc[0] > n_max_cluster
                # If the largest cluster is not too big, save clustering nad continue
                if not clustering_failed:
                    clusters_saved = clusters
                else:
                    break
            
            # Save the clusters
            if clusters_saved is not None:
                clusters_saved[clusters_saved == -1] = np.nan
                df.loc[df_identity.index, 'cluster'] = clusters_saved.to_numpy()

        return df['cluster'].to_numpy()

    def resplit_by_clusters(
            self,
            df: pd.DataFrame,
            clusters: np.ndarray,
            idx_train: np.ndarray,
            ) -> Tuple[np.ndarray, np.ndarray]:
        
        """Creates a random re-split of an already existing split.

        The re-split is based on clusters which collect similar images.
        Then it puts of similar images into the training set.
        The rest is randomly split into training and testing sets.
        The re-split mimics the split as the training set contains
        the same number of samples for EACH individual.
        The same goes for the testing set.
        
        Args:
            df (pd.DataFrame): A dataframe of the data. It must contain column `identity`.
            clusters (np.ndarray): An array of clusters with the same length as `df`.
            idx_train (np.ndarray): Labels of the training set.

        Returns:
            List of labels of the training and testing sets.
        """
        
        df = self.modify_df(df)

        # Replace clusters appearing just ones with np.nan
        clusters_unique, clusters_count = np.unique(clusters, return_counts=True)
        clusters[np.isin(clusters, clusters_unique[clusters_count == 1])] = np.nan
        df['cluster'] = clusters

        # Initialize the random number generator
        lcg = self.initialize_lcg()

        # Determine how many images of each individual should be in the training set
        identity_train_counts = df.loc[idx_train][self.col_label].value_counts()

        # Loop over individuals and create a split for each
        idx_train_new = []
        for identity, df_identity in tqdm(df.groupby(self.col_label), disable=self.disable_tqdm):
            n_train = identity_train_counts.get(identity, 0)
            if len(df_identity) - n_train <= 1:
                # All or all but one samples into the training set
                idx_remaining = np.array(df_identity.index)
                idx_remaining = lcg.random_shuffle(idx_remaining)
                idx_train_identity = idx_remaining[:n_train]
            else:
                # Add all the clusters into the training set
                idx_train_identity = []
                for _, df_cluster in df_identity.groupby('cluster'):
                    # Check if the training set is not too big
                    if len(idx_train_identity) + len(df_cluster) <= n_train:
                        idx_train_identity += list(df_cluster.index)

                # Distribute the remaining indices
                n_train_remaining = n_train - len(idx_train_identity)
                idx_remaining = self.setdiff(df_identity.index, idx_train_identity)
                idx_remaining = lcg.random_shuffle(idx_remaining)
                idx_train_identity += list(idx_remaining[:n_train_remaining])
            idx_train_new += list(idx_train_identity)
        idx_test_new = self.setdiff(df.index, idx_train_new)
        return np.array(idx_train_new), np.array(idx_test_new)

    def set_col_label(self, col_label: str) -> None:
        """Sets col_label to desired value

        Args:
            col_label (str): Desired value for col_label.
        """

        self.col_label = col_label

    def setdiff(self, a, b):
        a = np.array(a)
        b = np.array(b)
        return pd.unique(a[~np.in1d(a,b)])
