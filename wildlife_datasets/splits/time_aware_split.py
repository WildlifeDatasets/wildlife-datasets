import numpy as np
import pandas as pd
from typing import List, Tuple
from .balanced_split import BalancedSplit


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
        self.y_unique = np.sort(self.df['identity'].unique())
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
