import numpy as np
import pandas as pd
from typing import List, Tuple

# TODO: add documentation
class Lcg():
    '''        
    Simple Linear congruential generator for generating random numbers.
    Copied from https://stackoverflow.com/questions/18634079/glibc-rand-function-implementation        
    It is machine-, distribution- and package version-independent.
    It has some drawbacks (check the link above) but perfectly sufficient for our application.
    '''
    def __init__(self, seed: int, iterate: int=0) -> None:
        self.state = seed
        for _ in range(iterate):
            self.random()

    def random(self) -> int:
        '''
        Generate random integer from the current state.        
        '''
        self.state = (self.state * 1103515245 + 12345) & 0x7FFFFFFF
        return self.state

    def random_permutation(self, n: int) -> List[int]:
        '''
        Generate random permutation of range(n).
        '''
        rnd = []
        for _ in range(n):
            self.random()
            rnd.append(self.state)
        return np.argsort(rnd)

    def random_shuffle(self, x: np.ndarray) -> np.ndarray:
        return x[self.random_permutation(len(x))]



class BalancedSplit():
    '''
    Basic splitter class, which should implement the split() method.
    It should perform balanced splits separately for all classes.
    Its children are ClosedSetSplit, OpenSetSplit, DisjointSetSplit and TimeAwareSplit.
    TimeAwareSplit furher has children TimeProportionSplit and TimeCutoffSplit.
    '''
    def __init__(self, df: pd.DataFrame, seed: int, keep_unknown: bool=False) -> None:
        '''
        The split is based on the DataFrame df, which needs to contain the column identity.
        For TimeAwareSplit it also needs to contain the column date.
        It may remove the unknown identities based on the keyword argument keep_unknown.
        '''
        # Potentially remove the unknown identities
        self.df = df.copy()
        if keep_unknown:
            self.df = self.df
        else:
            self.df = self.df[self.df['identity'] != 'unknown']
        # Initialize the random number generator
        self.change_seed(seed)

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

    def change_seed(self, seed: int) -> None:
        '''
        Changes the seed of the random number generator.
        '''
        self.lcg = Lcg(seed)
    
    def split(self, *args, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        '''
        Splitting method which needs to be implemented by subclasses.

        It splits the DataFrame self.df into indices idx_train and idx_test.
        The subdataset is obtained by loc (not iloc), therefore self.df.loc[idx_train].
        '''
        raise(NotImplementedError('Subclasses should implement this. \n You may want to use ClosedSetSplit instead of BalancedSplit.'))

    def general_split(self, ratio_train, individual_train, individual_test,
        ratio_train_min: float=0, ratio_train_max: float=1
        ) -> Tuple[np.ndarray, np.ndarray]:
        '''
        For a general idea, see the documentation of the BalancedSplit.split() function.
        This split is a general purpose split with the following arguments:
            ratio_train is the !approximate! size of the training set. It is applied to each individual separately.
            individual_train are the names of individuals which go ALL to the training set.
            individual_test are the names of individuals which go ALL to the testing set.            
        Since some individuals may go only one of the sets, some recomputation of ratio_train is needed.
        The recomputed value will always lie in the interval [ratio_train_min, ratio_train_max].
        '''
        # Compute how many samples go automatically to the training and testing sets
        n_train = sum(self.y_counts[[k in individual_train for k in self.y_unique]])
        n_test = sum(self.y_counts[[k in individual_test for k in self.y_unique]])
        
        # Recompute ratio_train and adjust it to proper bounds 
        if n_train + n_test > 0 and n_train + n_test < self.n:
            ratio_train = (self.n*ratio_train - n_train) / (self.n - n_test - n_train)
        ratio_train = np.minimum(np.maximum(ratio_train, ratio_train_min), ratio_train_max)
        
        idx_train = np.empty(self.n, dtype='bool')
        # Make a loop over all individuals
        for individual, y_count in zip(self.y_unique, self.y_counts):            
            if individual in individual_train and individual in individual_test:
                # Check if the class does not belong to both sets
                raise(Exception('Individual cannot be both in individual_train and individual_test.'))
            elif individual in individual_train:
                # Check if the class does not belong to the training set
                idx_train_class = np.ones(y_count, dtype='bool')
            elif individual in individual_test:
                # Check if the class does not belong to the testing set
                idx_train_class = np.zeros(y_count, dtype='bool')
            else:
                idx_train_class = np.zeros(y_count, dtype='bool')
                # Otherwise compute the number of samples in the training set
                n_train = np.round(ratio_train * y_count).astype(int)
                if n_train == y_count and n_train > 1:
                    n_train -= 1
                if n_train == 0:
                    n_train = 1
                # Create indices to the training set and randomly permute them                
                idx_permutation = self.lcg.random_permutation(y_count)
                idx_train_class[:n_train] = True                
                idx_train_class = idx_train_class[idx_permutation]
            # Save the indices
            idx_train[self.y == individual] = idx_train_class
        return np.array(self.df.index.values)[idx_train], np.array(self.df.index.values)[~idx_train]


class ClosedSetSplit(BalancedSplit):
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


class OpenSetSplit(BalancedSplit):
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


class DisjointSetSplit(BalancedSplit):
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


class TimeAwareSplit(BalancedSplit):
    '''
    Time-aware splits are based on https://arxiv.org/abs/2211.10307
    The split is not created randomly but based on time from self.df['date'].
    This creates more complicated split that the time-unaware (random) one.
    '''
    def __init__(self, *args, **kwargs) -> None:
        '''
        For a general idea, see the documentation of the BalancedSplit.__init__() function.
        '''
        super().__init__(*args, **kwargs)
        if 'date' not in self.df.columns:
            # Check if the DataFrame contain the column date.
            raise(Exception('Dataframe df does not contain column date.'))
        # Convert date to datetime format (from possibly strings) and drop hours
        self.df['date'] = pd.to_datetime(self.df['date']).apply(lambda x: x.date())
        if 'year' not in self.df.columns:
            # Extract year from the date format
            self.df['year'] = self.df['date'].apply(lambda x: x.year).to_numpy()

    def resplit_random(self, idx_train, idx_test, year_max=np.inf):
        '''
        For a general idea, see the documentation of the BalancedSplit.split() function.
        This function creates a random re-split of an already generated split.
        The re-split mimics the split as the training set contains the same number of samples for EACH individual.
        The same goes for the testing set.
        The re-split samples may be drawn only from self.df['year'] <= year_max.
        '''
        # Compute the number of samples for each individual in the training set
        counts_train = {}
        for x in self.df.loc[idx_train].groupby('identity'):
            counts_train[x[0]] = len(x[1])
        # Compute the number of samples for each individual in the testing set
        counts_test = {}
        for x in self.df.loc[idx_test].groupby('identity'):
            counts_test[x[0]] = len(x[1])

        idx_train_new = []
        idx_test_new = []
        # Loop over all individuals
        for identity in self.y_unique:
            # Extract the number of individuals in the training and testing sets
            n_train = counts_train.get(identity, 0)
            n_test = counts_test.get(identity, 0)
            # Get randomly permuted indices of the corresponding identity
            idx = np.where(self.df['identity'] == identity)[0]
            idx = idx[self.df.iloc[idx]['year'] <= year_max]
            idx = self.lcg.random_shuffle(idx)
            if len(idx) < n_train+n_test:
                raise(Exception('The set is too small.'))
            # Get the correct number of indices in both sets
            idx_train_new += list(idx[:n_train])
            idx_test_new += list(idx[n_train:n_train+n_test])
        return np.array(self.df.index.values)[idx_train_new], np.array(self.df.index.values)[idx_test_new]


class TimeProportionSplit(TimeAwareSplit):    
    '''
    Time-proportion split is based on https://arxiv.org/abs/2211.10307
    For each individual, it extracts unique observation dates and puts half to the training to the testing set.    
    Ignores individuals with only one observation date.
    '''    
    def split(self) -> Tuple[np.ndarray, np.ndarray]:
        '''
        For a general idea, see the documentation of the BalancedSplit.split() function.
        '''
        idx_train = []
        idx_test = []
        # Loop over all identities; x is a tuple (identity, df with unique identity)
        for x in self.df.groupby('identity'):            
            dates = x[1].groupby('date')
            n_dates = len(dates)
            if n_dates > 1:
                # Loop over all dates; y is a tuple (date, df with unique date and identity)
                for i, y in enumerate(dates):
                    # Add half dates to the training and half to the testing set
                    if i < int(np.round(n_dates/2)):
                        idx_train += list(y[1].index)
                    else:
                        idx_test += list(y[1].index)
        return idx_train, idx_test


class TimeCutoffSplit(TimeAwareSplit):
    '''
    Time-cutoff split is based on https://arxiv.org/abs/2211.10307
    Puts all individuals observed before 'year' into the training test.
    Puts all individuals observed during 'year' into the testing test.
    Ignores all individuals observed after 'year'.
    '''     
    def split(self, year: int) -> Tuple[np.ndarray, np.ndarray]:
        '''
        For a general idea, see the documentation of the BalancedSplit.split() function.
        '''
        idx_train = list(np.where(self.df['year'] < year)[0])
        idx_test = list(np.where(self.df['year'] == year)[0])
        return np.array(self.df.index.values)[idx_train], np.array(self.df.index.values)[idx_test]

    def splits_all(self) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], np.ndarray]:
        '''
        Creates splits for all possible splitting years
        '''
        # Ignores the first year because otherwise the training set would be empty
        years = np.sort(self.df['year'].unique())[1:]
        splits = []
        for year in years:
            splits.append(self.split(year))
        return splits, years
