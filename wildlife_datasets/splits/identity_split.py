import numpy as np
from typing import List, Tuple
from .balanced_split import BalancedSplit


class IdentitySplit(BalancedSplit):
    def __init__(self, *args, **kwargs) -> None:
        '''
        For a general idea, see the documentation of the BalancedSplit.__init__() function.
        '''
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
    '''
    ClosedSetSplit is the split where all individuals are in the training and testing set.
    The only exception is that individuals with one sample are in the training set only.
    '''
    def split(self, ratio_train: float) -> Tuple[np.ndarray, np.ndarray]:
        '''
        For a general idea, see the documentation of the BalancedSplit.split() function.
        The size of the training set is approximately ratio_train.
        '''
        individual_train = np.array([], dtype=object)
        individual_test = np.array([], dtype=object)
        return self.general_split(ratio_train, individual_train, individual_test)


class OpenSetSplit(IdentitySplit):
    '''
    OpenSetSplit is the split where some individuals are in the testing but not in the training set.
    These are the newly observed individuals.
    '''
    def split(self, ratio_train, ratio_class_test: float=None, n_class_test: int=None) -> Tuple[np.ndarray, np.ndarray]:
        '''
        For a general idea, see the documentation of the BalancedSplit.split() function.
        The size of the training set is approximately ratio_train.
        The classes in the testing set can be determined by:
            ratio_class_test: !approximate! relative number of the samples (not individuals) in the testing set.
            n_class_test: absolute number of the individuals (classes) in the testing sets.
        '''
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
    '''
    DisjointSetSplit is the split where NO individual is both in the training and testing set.
    '''
    def split(self, ratio_class_test: float=None, n_class_test: int=None) -> Tuple[np.ndarray, np.ndarray]:
        '''
        For a general idea, see the documentation of the BalancedSplit.split() function.
        The classes in the testing set can be determined by:
            ratio_class_test: !approximate! relative number of the samples (not individuals) in the testing set.
            n_class_test: absolute number of the individuals (classes) in the testing sets.
        '''
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

