import numpy as np
import pandas as pd
from typing import List, Tuple
from .balanced_split import BalancedSplit


class ClosedSetSplit(BalancedSplit):
    """Closed-set splitting method into training and testing sets.

    All individuals are both in the training and testing set.
    The only exception is that individuals with only one sample are in the training set.
    Implementation of [this paper](https://arxiv.org/abs/2211.10307).
    """

    def __init__(
            self,
            ratio_train: float,
            seed: int = 666,
            identity_skip: str = 'unknown',
            ) -> None:
        """Initializes the class.

        Args:
            ratio_train (float): *Approximate* size of the training set.
            seed (int, optional): Initial seed for the LCG random generator.
            identity_skip (str, optional): Name of the identities to ignore.
        """

        self.ratio_train = ratio_train
        self.identity_skip = identity_skip
        self.set_seed(seed)
    
    def split(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Implementation of the [base splitting method](../reference_splits#splits.balanced_split.BalancedSplit.split).

        Args:
            df (pd.DataFrame): A dataframe of the data. It must contain column `identity`.

        Returns:
            List of labels of the training and testing sets.
        """

        df = df[df['identity'] != self.identity_skip]
        individual_train = np.array([], dtype=object)
        individual_test = np.array([], dtype=object)
        return self.general_split(df, individual_train, individual_test)


class OpenSetSplit(BalancedSplit):
    """Open-set splitting method into training and testing sets.

    Some individuals are in the testing but not in the training set.
    Implementation of [this paper](https://arxiv.org/abs/2211.10307).
    """

    def __init__(
            self,
            ratio_train: float,
            ratio_class_test: float = None,
            n_class_test: int = None,
            seed: int = 666,
            identity_skip: str = 'unknown',
            ) -> None:
        """Initializes the class.

        The user must provide exactly one from `ratio_class_test` and `n_class_test`.
        The latter specifies the number of individuals to be only in the testing set.
        The former specified the ratio of samples of individuals (not individuals themselves)
        to be only in the testing set.

        Args:
            ratio_train (float): *Approximate* size of the training set.
            ratio_class_test (float, optional): *Approximate* ratio of samples of individuals only in the testing set.
            n_class_test (int, optional): Number of individuals only in the testing set.
            seed (int, optional): Initial seed for the LCG random generator.
            identity_skip (str, optional): Name of the identities to ignore.
        """

        if ratio_class_test is None and n_class_test is None:
            raise(Exception('Either ratio_class_test or n_class_test must be provided.'))
        elif ratio_class_test is not None and n_class_test is not None:
            raise(Exception('Only ratio_class_test or n_class_test can be provided.'))
        
        self.ratio_train = ratio_train
        self.ratio_class_test = ratio_class_test
        self.n_class_test = n_class_test
        self.identity_skip = identity_skip
        self.set_seed(seed)

    def split(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Implementation of the [base splitting method](../reference_splits#splits.balanced_split.BalancedSplit.split).

        Args:
            df (pd.DataFrame): A dataframe of the data. It must contain column `identity`.

        Returns:
            List of labels of the training and testing sets.
        """
        
        # Remove identities to be skipped
        df = df[df['identity'] != self.identity_skip]
        
        # Compute the counts and randomly permute them
        y_counts = df['identity'].value_counts()
        n_class = len(y_counts)
        idx = self.lcg.random_permutation(n_class)
        y_counts = y_counts.iloc[idx]

        # Compute number of identities in the testing set
        n = len(df)
        if self.n_class_test is None:
            n_test = np.round(n*self.ratio_class_test).astype(int)
            n_class_test = np.where(np.cumsum(y_counts) >= n_test)[0][0]
        else:
            n_class_test = self.n_class_test

        # Specify individuals going purely into training and testing sets
        individual_train = np.array([], dtype=object)
        individual_test = np.array(y_counts.index[:n_class_test])
        return self.general_split(df, individual_train, individual_test)


class DisjointSetSplit(BalancedSplit):
    """Disjoint-set splitting method into training and testing sets.

    No individuals are in both the training and testing sets.
    Implementation of [this paper](https://arxiv.org/abs/2211.10307).
    """

    def __init__(
            self,
            ratio_class_test: float = None,
            n_class_test: int = None,
            seed: int = 666,
            identity_skip: str = 'unknown',
            ) -> None:
        """Initializes the class.

        The user must provide exactly one from `ratio_class_test` and `n_class_test`.
        The latter specifies the number of individuals to be only in the testing set.
        The former specified the ratio of samples of individuals (not individuals themselves)
        to be only in the testing set.

        Args:
            ratio_class_test (float, optional): *Approximate* ratio of samples of individuals only in the testing set.
            n_class_test (int, optional): Number of individuals only in the testing set.
            seed (int, optional): Initial seed for the LCG random generator.
            identity_skip (str, optional): Name of the identities to ignore.
        """

        if ratio_class_test is None and n_class_test is None:
            raise(Exception('Either ratio_class_test or n_class_test must be provided.'))
        elif ratio_class_test is not None and n_class_test is not None:
            raise(Exception('Only ratio_class_test or n_class_test can be provided.'))
        
        self.ratio_train = 0 # Arbitrary value
        self.ratio_class_test = ratio_class_test
        self.n_class_test = n_class_test
        self.identity_skip = identity_skip
        self.set_seed(seed)

    def split(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Implementation of the [base splitting method](../reference_splits#splits.balanced_split.BalancedSplit.split).

        Args:
            df (pd.DataFrame): A dataframe of the data. It must contain column `identity`.
        
        Returns:
            List of labels of the training and testing sets.
        """

        # Remove identities to be skipped
        df = df[df['identity'] != self.identity_skip]
        
        # Compute the counts and randomly permute them
        y_counts = df['identity'].value_counts()
        n_class = len(y_counts)
        idx = self.lcg.random_permutation(n_class)
        y_counts = y_counts.iloc[idx]

        # Compute number of identities in the testing set
        n = len(df)
        if self.n_class_test is None:
            n_test = np.round(n*self.ratio_class_test).astype(int)
            n_class_test = np.where(np.cumsum(y_counts) >= n_test)[0][0]
        else:
            n_class_test = self.n_class_test

        # Specify individuals going purely into training and testing sets
        individual_train = np.array(y_counts.index[n_class_test:])
        individual_test = np.array(y_counts.index[:n_class_test])
        return self.general_split(df, individual_train, individual_test)

