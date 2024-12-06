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

    def resplit_by_features(
            self,
            df: pd.DataFrame,
            features: np.ndarray,
            idx_train: np.ndarray,
            n_max_cluster: int = 5,
            eps_min: float = 0.01,
            eps_max: float = 0.50,
            eps_step: float = 0.01,
            min_samples: int = 2,
            save_clusters_prefix: Optional[str] = None,
            ) -> Tuple[np.ndarray, np.ndarray]:
        
        """Creates a random re-split of an already existing split.

        The re-split is based on similarity of features.
        It runs DBSCAN with increasing eps (cluster radius) until
        the clusters are smaller than `n_max_cluster`.
        Then it puts of similar images into the training set.
        The rest is randomly split into training and testing sets.
        The re-split mimics the split as the training set contains
        the same number of samples for EACH individual.
        The same goes for the testing set.
        
        Args:
            df (pd.DataFrame): A dataframe of the data. It must contain column `identity`.
            features (np.ndarray): An array of features with the same length as `df`.
            idx_train (np.ndarray): Labels of the training set.
            n_max_cluster (int, optional): Maximal size of cluster before `eps` stops increasing.
            eps_min (float, optional): Lower bound for epsilon.
            eps_max (float, optional): Upper bound for epsilon.
            eps_step (float, optional): Step for epsilon.
            min_samples (int, optional): Minimal cluster size.
            save_clusters_prefix (Optional[bool], optional): File name prefix for saving clusters.

        Returns:
            List of labels of the training and testing sets.
        """
        
        # Modify the dataframe if the function is present
        if hasattr(self, 'modify_df'):
            df = self.modify_df(df)

        # Initialize the random number generator
        lcg = self.initialize_lcg()

        # Determine how many images of each individual should be in the training set
        identity_train_counts = df.loc[idx_train][self.col_label].value_counts()

        # Loop over individuals and create a split for each
        idx_train_new = []
        for identity, df_identity in tqdm(df.groupby(self.col_label)):
            n_train = identity_train_counts.get(identity, 0)
            if len(df_identity) - n_train <= 1:
                # All or all but one samples into the training set
                idx_remaining = np.array(df_identity.index)
                idx_remaining = lcg.random_shuffle(idx_remaining)
                idx_train_identity = idx_remaining[:n_train]
            else:
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
                if save_clusters_prefix is not None:
                    df_save = pd.DataFrame({'cluster': clusters_saved.to_numpy()}, index=df_identity.index)
                    df_save.to_csv(f'{save_clusters_prefix}_{identity}.csv')
                
                # Add all the clusters into the training set
                idx_train_identity = []
                if clusters_saved is not None:
                    for cluster, df_cluster in pd.DataFrame({'cluster': clusters_saved}).groupby('cluster'):
                        # Check if the training set is not too big
                        if cluster != -1 and len(idx_train_identity) + len(df_cluster) <= n_train:
                            idx_train_identity += list(df_identity.index[df_cluster.index])

                # Distribute the remaining indices
                n_train_remaining = n_train - len(idx_train_identity)
                idx_remaining = np.array(list(set(df_identity.index) - set(idx_train_identity)))
                idx_remaining = lcg.random_shuffle(idx_remaining)
                idx_train_identity += list(idx_remaining[:n_train_remaining])
            idx_train_new += list(idx_train_identity)
        idx_test_new = list(set(df.index) - set(idx_train_new))
        return np.array(idx_train_new), np.array(idx_test_new)

    def set_col_label(self, col_label: str) -> None:
        """Sets col_label to desired value

        Args:
            col_label (str): Desired value for col_label.
        """

        self.col_label = col_label
