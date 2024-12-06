import numpy as np
import pandas as pd
from typing import List, Tuple
from .balanced_split import BalancedSplit


class FullSplit(BalancedSplit):
    """Simplest split returning the whole dataset.
    """

    def __init__(self):
        pass

    def split(self, df: pd.DataFrame) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Implementation of the [base splitting method](../reference_splits#splits.balanced_split.BalancedSplit.split).

        Args:
            df (pd.DataFrame): A dataframe of the data. It must contain column `identity`.

        Returns:
            List of splits. Each split is list of labels of the training and testing sets.
        """

        return [ (df.index.values, np.array([], dtype=int)) ]
    

class IdentitySplit(BalancedSplit):
    """Base class for `ClosedSetSplit`, `OpenSetSplit` and `DisjointSetSplit`.
    """

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
    
    def general_split(
            self,
            df: pd.DataFrame,
            individual_train: List[str],
            individual_test: List[str],
            ) -> Tuple[np.ndarray, np.ndarray]:
        """General-purpose split into the training and testing sets.

        It puts all samples of `individual_train` into the training set
        and all samples of `individual_test` into the testing set.
        The splitting is performed for each individual separately.
        The split will result in at least one sample in both the training and testing sets.
        If only one sample is available for an individual, it will be in the training set.
                
        Args:
            df (pd.DataFrame): A dataframe of the data. It must contain column `identity`.
            individual_train (List[str]): Individuals to be only in the training test.
            individual_test (List[str]): Individuals to be only in the testing test.

        Returns:
            List of labels of the training and testing sets.
        """
        
        # Initialize the random number generator
        lcg = self.initialize_lcg()

        # Compute how many samples go automatically to the training and testing sets
        y_counts = df[self.col_label].value_counts()
        n_train = sum([y_counts.loc[y] for y in individual_train])
        n_test = sum([y_counts.loc[y] for y in individual_test])
        
        # Recompute ratio_train and adjust it to proper bounds
        n = len(df)
        ratio_train = self.ratio_train
        if n_train + n_test > 0 and n_train + n_test < n:
            ratio_train = (n*ratio_train - n_train) / (n - n_test - n_train)
        ratio_train = np.clip(ratio_train, 0, 1)

        idx_train = []
        idx_test = []        
        # Make a loop over all individuals
        for individual, df_individual in df.groupby(self.col_label):
            if individual in individual_train and individual in individual_test:
                # Check if the class does not belong to both sets
                raise(Exception('Individual cannot be both in individual_train and individual_test.'))
            elif individual in individual_train:
                # Check if the class does not belong to the training set
                idx_train += list(df_individual.index)
            elif individual in individual_test:
                # Check if the class does not belong to the testing set
                idx_test += list(df_individual.index)
            else:
                # Otherwise compute the number of samples in the training set
                n_individual = len(df_individual)
                n_train = np.round(ratio_train * n_individual).astype(int)
                if n_train == n_individual and n_train > 1:
                    n_train -= 1
                if n_train == 0:
                    n_train = 1
                # Create indices to the training set and randomly permute them                
                idx_permutation = lcg.random_permutation(n_individual)
                idx_permutation = np.array(idx_permutation)
                idx_train += list(df_individual.index[idx_permutation[:n_train]])
                idx_test += list(df_individual.index[idx_permutation[n_train:]])
        return np.array(idx_train), np.array(idx_test)
    

class ClosedSetSplit(IdentitySplit):
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
            col_label: str = 'identity',            
            ) -> None:
        """Initializes the class.

        Args:
            ratio_train (float): *Approximate* size of the training set.
            seed (int, optional): Initial seed for the LCG random generator.
            identity_skip (str, optional): Name of the identities to ignore.
            col_label (str, optional): Column name containing individual animal names (labels).
        """

        self.ratio_train = ratio_train
        self.identity_skip = identity_skip
        self.seed = seed
        self.col_label = col_label
    
    def split(self, df: pd.DataFrame) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Implementation of the [base splitting method](../reference_splits#splits.balanced_split.BalancedSplit.split).

        Args:
            df (pd.DataFrame): A dataframe of the data. It must contain column `identity`.

        Returns:
            List of splits. Each split is list of labels of the training and testing sets.
        """

        df = self.modify_df(df)
        individual_train = np.array([], dtype=object)
        individual_test = np.array([], dtype=object)
        return [self.general_split(df, individual_train, individual_test)]


class OpenSetSplit(IdentitySplit):
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
            col_label: str = 'identity',
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
            col_label (str, optional): Column name containing individual animal names (labels).
        """

        if ratio_class_test is None and n_class_test is None:
            raise(Exception('Either ratio_class_test or n_class_test must be provided.'))
        elif ratio_class_test is not None and n_class_test is not None:
            raise(Exception('Only ratio_class_test or n_class_test can be provided.'))
        
        self.ratio_train = ratio_train
        self.ratio_class_test = ratio_class_test
        self.n_class_test = n_class_test
        self.identity_skip = identity_skip
        self.seed = seed
        self.col_label = col_label

    def split(self, df: pd.DataFrame) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Implementation of the [base splitting method](../reference_splits#splits.balanced_split.BalancedSplit.split).

        Args:
            df (pd.DataFrame): A dataframe of the data. It must contain column `identity`.

        Returns:
            List of splits. Each split is list of labels of the training and testing sets.
        """
        
        df = self.modify_df(df)

        # Initialize the random number generator
        lcg = self.initialize_lcg()

        # Compute the counts and randomly permute them
        y_counts = df[self.col_label].value_counts()
        n_class = len(y_counts)
        idx = lcg.random_permutation(n_class)
        y_counts = y_counts.iloc[idx]

        # Compute number of identities in the testing set
        n = len(df)
        if self.n_class_test is None:
            n_test = np.round(n*self.ratio_class_test).astype(int)
            n_class_test = np.where(np.cumsum(y_counts) >= n_test)[0][0] + 1
        else:
            n_class_test = self.n_class_test

        # Specify individuals going purely into training and testing sets
        individual_train = np.array([], dtype=object)
        individual_test = np.array(y_counts.index[:n_class_test])
        return [self.general_split(df, individual_train, individual_test)]


class DisjointSetSplit(IdentitySplit):
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
            col_label: str = 'identity',
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
            col_label (str, optional): Column name containing individual animal names (labels).            
        """

        if ratio_class_test is None and n_class_test is None:
            raise(Exception('Either ratio_class_test or n_class_test must be provided.'))
        elif ratio_class_test is not None and n_class_test is not None:
            raise(Exception('Only ratio_class_test or n_class_test can be provided.'))
        
        self.ratio_train = 0 # Arbitrary value
        self.ratio_class_test = ratio_class_test
        self.n_class_test = n_class_test
        self.identity_skip = identity_skip
        self.seed = seed
        self.col_label = col_label

    def split(self, df: pd.DataFrame) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Implementation of the [base splitting method](../reference_splits#splits.balanced_split.BalancedSplit.split).

        Args:
            df (pd.DataFrame): A dataframe of the data. It must contain column `identity`.
        
        Returns:
            List of splits. Each split is list of labels of the training and testing sets.
        """

        df = self.modify_df(df)
        
        # Initialize the random number generator
        lcg = self.initialize_lcg()

        # Compute the counts and randomly permute them
        y_counts = df[self.col_label].value_counts()
        n_class = len(y_counts)
        idx = lcg.random_permutation(n_class)
        y_counts = y_counts.iloc[idx]

        # Compute number of identities in the testing set
        n = len(df)
        if self.n_class_test is None:
            n_test = np.round(n*self.ratio_class_test).astype(int)
            n_class_test = np.where(np.cumsum(y_counts) >= n_test)[0][0] + 1
        else:
            n_class_test = self.n_class_test

        # Specify individuals going purely into training and testing sets
        individual_train = np.array(y_counts.index[n_class_test:])
        individual_test = np.array(y_counts.index[:n_class_test])
        return [self.general_split(df, individual_train, individual_test)]

