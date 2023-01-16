import numpy as np
import pandas as pd
from typing import List


class Lcg():
    def __init__(self, seed: int, iterate: int = 0):
        '''        
        Simple Linear congruential generator for generating random numbers.
        Copied from https://stackoverflow.com/questions/18634079/glibc-rand-function-implementation        
        It is machine-, distribution- and package version-independent.
        It has some drawbacks (check the link above) but perfectly sufficient for our application.
        '''
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


class BalancedSplit():
    def __init__(self, df, seed, keep_unknown=False):
        self.df = df.copy()
        if keep_unknown:
            self.df = self.df
        else:
            self.df = self.df[self.df['identity'] != 'unknown']
        self.lcg = Lcg(seed)

        y = self.df['identity'].to_numpy()                
        _, y_idx, y_counts = np.unique(y, return_index=True, return_counts=True)
        y_unique = np.array([y[index] for index in sorted(y_idx)])
        y_counts = y_counts[np.argsort(y_idx)]

        self.y = y
        self.y_unique = y_unique
        self.y_counts = y_counts
        self.n = len(y)
        self.n_class = len(y_unique)

    def split(self, *args, **kwargs):
        raise(NotImplementedError('Subclasses should implement this. \n You may want to use ClosedSetSplit instead of BalancedSplit.'))

    def general_split(self, ratio_train, individual_train, individual_test, ratio_train_min=0, ratio_train_max=1):
        # check if the intersection of individuals to the train and test sets is empty
        if np.intersect1d(individual_train, individual_test).size > 0:
            raise(Exception('The intersection of individual_train and individual_test must be empty.'))
        
        # how many samples goes automatically to the train and test sets
        n_train = sum(self.y_counts[[k in individual_train for k in self.y_unique]])
        n_test = sum(self.y_counts[[k in individual_test for k in self.y_unique]])
        
        # recompute p and adjust it to proper bounds 
        if n_train + n_test > 0 and n_train + n_test < self.n:
            ratio_train = (self.n*ratio_train - n_train) / (self.n - n_test - n_train)
        ratio_train = np.minimum(np.maximum(ratio_train, ratio_train_min), ratio_train_max)
        
        idx_train = np.empty(self.n, dtype='bool')
        for individual, y_count in zip(self.y_unique, self.y_counts):            
            # check if the class does not go fully to the train or test set
            if individual in individual_train:
                idx_train_class = np.ones(y_count, dtype='bool')
            elif individual in individual_test:
                idx_train_class = np.zeros(y_count, dtype='bool')
            else:
                idx_train_class = np.zeros(y_count, dtype='bool')

                # number of samples to the train set
                n_train = np.round(ratio_train * y_count).astype(int)
                if n_train == y_count and n_train > 1:
                    n_train -= 1
                if n_train == 0:
                    n_train = 1

                # create indices and randomly permute them                
                idx_permutation = self.lcg.random_permutation(y_count)
                idx_train_class[:n_train] = True                
                idx_train_class = idx_train_class[idx_permutation]

            idx_train[self.y == individual] = idx_train_class
        return np.array(self.df.index.values)[idx_train], np.array(self.df.index.values)[~idx_train]


class ClosedSetSplit(BalancedSplit):
    def split(self, ratio_train):
        individual_train = np.array([], dtype=object)
        individual_test = np.array([], dtype=object)
        return self.general_split(ratio_train, individual_train, individual_test)


class OpenSetSplit(BalancedSplit):
    def split(self, ratio_train, ratio_class_test):
        idx_permuted = self.lcg.random_permutation(self.n_class)
        y_unique = self.y_unique[idx_permuted]
        y_counts = self.y_counts[idx_permuted]        
        i_end = np.where(np.cumsum(y_counts) >= np.round(ratio_class_test * self.n).astype(int))[0][0]
        individual_train = np.array([], dtype=object)
        individual_test = np.array(y_unique[:i_end])
        return self.general_split(ratio_train, individual_train, individual_test)


class DisjointSetSplit(BalancedSplit):
    def split(self, ratio_class_test):
        idx_permuted = self.lcg.random_permutation(self.n_class)
        y_unique = self.y_unique[idx_permuted]
        y_counts = self.y_counts[idx_permuted]        
        i_end = np.where(np.cumsum(y_counts) >= np.round(ratio_class_test * self.n).astype(int))[0][0]
        individual_train = np.array(y_unique[i_end:])
        individual_test = np.array(y_unique[:i_end])
        return self.general_split([], individual_train, individual_test)


class TimeAwareSplit(BalancedSplit):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if 'date' not in self.df.columns:
            raise(Exception('Dataframe df does not contain column date.'))
        elif self.df['date'].dtypes == object:
            raise(Exception('Dataframe column df is of type object. \n Either it is string and then convert it to datetime. \n Or there are missing values and then remove them.'))
        if 'year' not in self.df.columns:
            self.df['year'] = self.df['date'].apply(lambda x: x.year).to_numpy()

    def resplit_random(self, split, year_max=np.inf):
        np.random.seed(self.seed)
        
        idx_train, idx_test = split
        counts_train = {}
        for x in self.df.iloc[idx_train].groupby('identity'):
            counts_train[x[0]] = len(x[1])
        counts_test = {}
        for x in self.df.iloc[idx_test].groupby('identity'):
            counts_test[x[0]] = len(x[1])

        idx_train_new = []
        idx_test_new = []
        for identity in self.df['identity'].unique():
            n_train = counts_train.get(identity, 0)
            n_test = counts_test.get(identity, 0)
            idx = np.where(self.df['identity'] == identity)[0]
            idx = idx[self.df.iloc[idx]['year'] <= year_max]
            idx = np.random.permutation(idx)
            if len(idx) < n_train+n_test:
                raise(Exception('The set is too small.'))
            idx_train_new += list(idx[:n_train])
            idx_test_new += list(idx[n_train:n_train+n_test])
        return idx_train_new, idx_test_new


class TimeProportionSplit(TimeAwareSplit):    
    def split(self):
        # TODO: some argument should be here
        idx_train = []
        idx_test = []
        for x in self.df.groupby('identity'):
            dates = x[1].groupby('date')
            n_dates = len(dates)
            if n_dates > 1:
                for i, y in enumerate(dates):
                    if i < int(np.round(n_dates/2)):
                        idx_train += list(y[1].index)
                    else:
                        idx_test += list(y[1].index)
        return idx_train, idx_test


class TimeCutoffSplit(TimeAwareSplit):
    def split(self):
        years = np.sort(self.df['year'].unique())[1:]
        splits = []
        for year in years:
            idx_train = list(np.where(self.df['year'] < year)[0])
            idx_test = list(np.where(self.df['year'] == year)[0])
            splits.append((idx_train, idx_test))
        return splits, years


class ReplicableRandomSplit:
    def __init__(self, splitter=ClosedSetSplit, n_splits=1, random_state=0, **kwargs):
        self.splitter = splitter
        self.n_splits = n_splits
        self.random_state = random_state
        self.kwargs = kwargs

    def split(self, indices, labels):
        splits = []
        for i in range(self.n_splits):
            df = pd.DataFrame({'identity': labels})
            splitter = self.splitter(df, self.random_state + i)
            split = splitter.split(**self.kwargs)
            splits.append( (indices[split[0]], indices[split[1]]) )
        return splits

