import numpy as np
import pandas as pd
from typing import List, Tuple
from .balanced_split import BalancedSplit


class TimeAwareSplit(BalancedSplit):
    """Base class for `TimeProportionSplit` and `TimeCutoffSplit`.
    """

    def __init__(self, *args, **kwargs) -> None:
        """Initializes the class with the same arguments as its [parent contructor](../reference_splits#splits.balanced_split.BalancedSplit.__init__).
        """      

        super().__init__(*args, **kwargs)
        if 'date' not in self.df.columns:
            # Check if the DataFrame contain the column date.
            raise(Exception('Dataframe df does not contain column date.'))
        # Removes entries without dates
        self.df = self.df[~self.df['date'].isnull()]
        # Convert date to datetime format (from possibly strings) and drop hours
        self.df['date'] = pd.to_datetime(self.df['date']).apply(lambda x: x.date())
        self.df['year'] = self.df['date'].apply(lambda x: x.year).to_numpy()            
        # Extract unique inidividuals
        self.y_unique = np.sort(self.df['identity'].unique())

    def resplit_random(
            self,
            idx_train: np.ndarray,
            idx_test: np.ndarray,
            year_max: int = np.inf
            ) -> Tuple[np.ndarray, np.ndarray]:
        """Creates a random re-split of an already existing split.

        The re-split mimics the split as the training set contains
        the same number of samples for EACH individual.
        The same goes for the testing set.
        The re-split samples may be drawn only from `self.df['year'] <= year_max`.

        Args:
            idx_train (np.ndarray): Labels of the training set.
            idx_test (np.ndarray): Labels of the testing set.
            year_max (int, optional): Considers only entries with `self.df['year'] <= year_max`.

        Returns:
            List of labels of the training and testing sets.
        """

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
            if n_train+n_test > 0:
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
    """Time-proportion non-random splitting method into training and testing sets.

    For each individual, it extracts unique observation dates
    and puts half to the training to the testing set.
    Ignores individuals with only one observation date.
    Implementation of [this paper](https://arxiv.org/abs/2211.10307).
    """

    def split(self) -> Tuple[np.ndarray, np.ndarray]:
        """Implementation of the [base splitting method](../reference_splits#splits.balanced_split.BalancedSplit.split).

        Returns:
            List of labels of the training and testing sets.
        """
        
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
    """Time-cutoff non-random splitting method into training and testing sets.

    Puts all individuals observed before `year` into the training test.
    Puts all individuals observed during `year` into the testing test.
    Ignores all individuals observed after `year`.
    Implementation of [this paper](https://arxiv.org/abs/2211.10307).
    """

    def split(self, year: int, test_one_year_only: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Implementation of the [base splitting method](../reference_splits#splits.balanced_split.BalancedSplit.split).

        Args:
            year (int): Splitting year.
            test_one_year_only (exact, optional): Whether the test set is `self.df['year'] == year` or `self.df['year'] >= year`.

        Returns:
            List of labels of the training and testing sets.
        """

        idx_train = list(np.where(self.df['year'] < year)[0])
        if test_one_year_only:
            idx_test = list(np.where(self.df['year'] == year)[0])
        else:
            idx_test = list(np.where(self.df['year'] >= year)[0])
        return np.array(self.df.index.values)[idx_train], np.array(self.df.index.values)[idx_test]

    def splits_all(self, **kwargs) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], np.ndarray]:
        """Creates `TimeCutoffSplit` splits for all possible splitting years.

        Returns:
            List of lists of labels of the training and testing sets.
            List of splitting years.
        """

        # Ignores the first year because otherwise the training set would be empty
        years = np.sort(self.df['year'].unique())[1:]
        splits = []
        for year in years:
            splits.append(self.split(year, **kwargs))
        return splits, years
