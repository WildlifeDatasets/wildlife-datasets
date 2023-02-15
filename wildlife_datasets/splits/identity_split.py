import numpy as np
import pandas as pd
from typing import List, Tuple
from .balanced_split import BalancedSplit


class IdentitySplit(BalancedSplit):
    """Base class for `ClosedSetSplit`, `OpenSetSplit` and `DisjointSetSplit`.
    """

    def __init__(self, *args, **kwargs) -> None:
        """Initializes the class with the same arguments as its [parent contructor](../reference_splits#splits.balanced_split.BalancedSplit.__init__).
        """      

        super().__init__(*args, **kwargs)
        # Compute unique classes (y_unique) and their counts (y_counts)
        y = self.df['identity'].to_numpy()                
        _, y_idx, y_counts = np.unique(y, return_index=True, return_counts=True)
        y_unique = np.array([y[index] for index in sorted(y_idx)])
        y_counts = y_counts[np.argsort(y_idx)]

        # Save the precomputed data
        self.y = y
        self.y_unique = y_unique
        self.y_counts = y_counts
        self.n = len(y)
        self.n_class = len(y_unique)


class ClosedSetSplit(IdentitySplit):
    """Closed-set splitting method into training and testing sets.

    All individuals are both in the training and testing set.
    The only exception is that individuals with only one sample are in the training set.
    Implementation of [this paper](https://arxiv.org/abs/2211.10307).
    """

    def split(self, ratio_train: float) -> Tuple[np.ndarray, np.ndarray]:
        """Implementation of the [base splitting method](../reference_splits#splits.balanced_split.BalancedSplit.split).

        Args:
            ratio_train (float): *Approximate* size of the training set.

        Returns:
            List of labels of the training and testing sets.
        """

        individual_train = np.array([], dtype=object)
        individual_test = np.array([], dtype=object)
        return self.general_split(ratio_train, individual_train, individual_test)


class OpenSetSplit(IdentitySplit):
    """Open-set splitting method into training and testing sets.

    Some individuals are in the testing but not in the training set.
    Implementation of [this paper](https://arxiv.org/abs/2211.10307).
    """
    
    def split(
            self,
            ratio_train: float,
            ratio_class_test: float = None,
            n_class_test: int = None
            ) -> Tuple[np.ndarray, np.ndarray]:
        """Implementation of the [base splitting method](../reference_splits#splits.balanced_split.BalancedSplit.split).

        The user must provide exactly one from `ratio_class_test` and `n_class_test`.
        The latter specifies the number of individuals to be only in the testing set.
        The former specified the ratio of samples of individuals (not individuals themselves)
        to be only in the testing set.

        Args:
            ratio_train (float): *Approximate* size of the training set.
            ratio_class_test (float, optional): *Approximate* ratio of samples of individuals only in the testing set.
            n_class_test (int, optional): Number of individuals only in the testing set.

        Returns:
            List of labels of the training and testing sets.
        """
        
        if ratio_class_test is None and n_class_test is None:
            raise(Exception('Either ratio_class_test or n_class_test must be provided.'))
        elif ratio_class_test is not None and n_class_test is not None:
            raise(Exception('Only ratio_class_test or n_class_test can be provided.'))
        else:
            # Randomly permute the counts
            idx_permuted = self.lcg.random_permutation(self.n_class)
            y_unique = self.y_unique[idx_permuted]
            y_counts = self.y_counts[idx_permuted]
            # If n_class_test is not provided, compute it from ratio_class_test
            if n_class_test is None:
                n_class_test = np.where(np.cumsum(y_counts) >= np.round(ratio_class_test * self.n).astype(int))[0][0]
        # Specify individuals going purely into training and testing sets
        individual_train = np.array([], dtype=object)
        individual_test = np.array(y_unique[:n_class_test])
        return self.general_split(ratio_train, individual_train, individual_test)


class DisjointSetSplit(IdentitySplit):
    """Disjoint-set splitting method into training and testing sets.

    No individuals are in both the training and testing sets.
    Implementation of [this paper](https://arxiv.org/abs/2211.10307).
    """

    def split(
            self,
            ratio_class_test: float = None,
            n_class_test: int = None
            ) -> Tuple[np.ndarray, np.ndarray]:
        """Implementation of the [base splitting method](../reference_splits#splits.balanced_split.BalancedSplit.split).

        The user must provide exactly one from `ratio_class_test` and `n_class_test`.
        The latter specifies the number of individuals to be only in the testing set.
        The former specified the ratio of samples of individuals (not individuals themselves)
        to be only in the testing set.

        Args:
            ratio_class_test (float, optional): *Approximate* ratio of samples of individuals only in the testing set.
            n_class_test (int, optional): Number of individuals only in the testing set.

        Returns:
            List of labels of the training and testing sets.
        """
        
        if ratio_class_test is None and n_class_test is None:
            raise(Exception('Either ratio_class_test or n_class_test must be provided.'))
        elif ratio_class_test is not None and n_class_test is not None:
            raise(Exception('Only ratio_class_test or n_class_test can be provided.'))
        else:
            # Randomly permute the counts        
            idx_permuted = self.lcg.random_permutation(self.n_class)
            y_unique = self.y_unique[idx_permuted]
            y_counts = self.y_counts[idx_permuted]
            # If n_class_test is not provided, compute it from ratio_class_test
            if n_class_test is None:
                n_class_test = np.where(np.cumsum(y_counts) >= np.round(ratio_class_test * self.n).astype(int))[0][0]
        # Specify individuals going purely into training and testing sets
        individual_train = np.array(y_unique[n_class_test:])
        individual_test = np.array(y_unique[:n_class_test])
        return self.general_split([], individual_train, individual_test)

