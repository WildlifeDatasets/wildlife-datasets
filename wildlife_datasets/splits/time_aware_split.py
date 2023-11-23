import numpy as np
import pandas as pd
from typing import List, Tuple
from .balanced_split import BalancedSplit


class TimeAwareSplit(BalancedSplit):
    """Base class for `TimeProportionSplit` and `TimeCutoffSplit`.
    """

    def modify_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepares dataframe for splits.

        Removes identities specified in `self.identity_skip` (usually unknown identities).
        Convert the `date` column into a unified format.
        Add the `year` column.

        Args:
            df (pd.DataFrame): A dataframe of the data. It must contain columns `identity` and `date`.

        Returns:
            Modified dataframe of the data.
        """
        
        # Check if the DataFrame contain the column date.
        if 'date' not in df.columns:
            raise(Exception('Dataframe df does not contain column date.'))
        
        # Remove identities to be skipped
        df = df.copy()
        df = df[df['identity'] != self.identity_skip]

        # Removes entries without dates
        df = df[~df['date'].isnull()]
        
        # Convert date to datetime format (from possibly strings) and drop hours
        df['date'] = pd.to_datetime(df['date']).apply(lambda x: x.date())
        df['year'] = df['date'].apply(lambda x: x.year).to_numpy()            
        return df

    def resplit_random(
            self,
            df: pd.DataFrame,
            idx_train: np.ndarray,
            idx_test: np.ndarray,
            year_max: int = np.inf
            ) -> Tuple[np.ndarray, np.ndarray]:
        """Creates a random re-split of an already existing split.

        The re-split mimics the split as the training set contains
        the same number of samples for EACH individual.
        The same goes for the testing set.
        The re-split samples may be drawn only from `df['year'] <= year_max`.

        Args:
            df (pd.DataFrame): A dataframe of the data. It must contain columns `identity` and `date`.
            idx_train (np.ndarray): Labels of the training set.
            idx_test (np.ndarray): Labels of the testing set.
            year_max (int, optional): Considers only entries with `df['year'] <= year_max`.

        Returns:
            List of labels of the training and testing sets.
        """

        df = self.modify_df(df)

        # Initialize the random number generator
        lcg = self.initialize_lcg()
        
        # Compute the number of samples for each individual in the training set
        counts_train = {}
        for name, df_name in df.loc[idx_train].groupby('identity'):
            counts_train[name] = len(df_name)
        # Compute the number of samples for each individual in the testing set
        counts_test = {}
        for name, df_name in df.loc[idx_test].groupby('identity'):
            counts_test[name] = len(df_name)

        idx_train_new = []
        idx_test_new = []
        # Loop over all individuals
        for name, df_name in df.groupby('identity'):
            # Extract the number of individuals in the training and testing sets
            n_train = counts_train.get(name, 0)
            n_test = counts_test.get(name, 0)
            if n_train+n_test > 0:
                # Get randomly permuted indices of the corresponding identity
                df_name = df_name[df_name['year'] <= year_max]
                if len(df_name) < n_train+n_test:
                    raise(Exception('The set is too small.'))
                # Get the correct number of indices in both sets
                idx_permutation = lcg.random_permutation(n_train+n_test)
                idx_permutation = np.array(idx_permutation)
                idx_train_new += list(df_name.index[idx_permutation[:n_train]])
                idx_test_new += list(df_name.index[idx_permutation[n_train:n_train+n_test]])
        return np.array(idx_train_new), np.array(idx_test_new)


class TimeProportionSplit(TimeAwareSplit):
    """Time-proportion non-random splitting method into training and testing sets.

    For each individual, it extracts unique observation dates
    and puts half to the training to the testing set.
    Ignores individuals with only one observation date.
    Implementation of [this paper](https://arxiv.org/abs/2211.10307).
    """

    def __init__(
            self,
            ratio: float = 0.5,
            seed: int = 666,
            identity_skip: str = 'unknown',
            ):
        """Initializes the class.

        Args:
            ratio (float, optional): The fraction of dates going to the testing set.
            seed (int, optional): Initial seed for the LCG random generator.
            identity_skip (str, optional): Name of the identities to ignore.
        """

        self.ratio = ratio
        self.identity_skip = identity_skip
        self.seed = seed

    def split(self, df: pd.DataFrame) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Implementation of the [base splitting method](../reference_splits#splits.balanced_split.BalancedSplit.split).

        Args:
            df (pd.DataFrame): A dataframe of the data. It must contain columns `identity` and `date`.

        Returns:
            List of splits. Each split is list of labels of the training and testing sets.
        """
        
        df = self.modify_df(df)
        idx_train = []
        idx_test = []
        # Loop over all identities; x is a tuple (identity, df with unique identity)
        for _, df_name in df.groupby('identity'):            
            dates = df_name.groupby('date')
            n_dates = len(dates)
            if n_dates > 1:
                # Loop over all dates; y is a tuple (date, df with unique date and identity)
                for i, (_, df_date) in enumerate(dates):
                    # Add half dates to the training and half to the testing set
                    if i < int(np.round(n_dates/2)):
                        idx_train += list(df_date.index)
                    else:
                        idx_test += list(df_date.index)
        return [(np.array(idx_train), np.array(idx_test))]


class TimeCutoffSplit(TimeAwareSplit):
    """Time-cutoff non-random splitting method into training and testing sets.

    Puts all individuals observed before `year` into the training test.
    Puts all individuals observed during `year` into the testing test.
    Ignores all individuals observed after `year`.
    Implementation of [this paper](https://arxiv.org/abs/2211.10307).
    """

    def __init__(
            self,
            year: int,
            test_one_year_only: bool = True,
            seed: int = 666,
            identity_skip: str = 'unknown',
            ) -> None:
        """Initializes the class.

        Args:
            year (int): Splitting year.
            test_one_year_only (bool, optional): Whether the test set is `df['year'] == year` or `df['year'] >= year`.
            seed (int, optional): Initial seed for the LCG random generator.            
            identity_skip (str, optional): Name of the identities to ignore.
        """

        self.year = year
        self.test_one_year_only = test_one_year_only
        self.identity_skip = identity_skip
        self.seed = seed
    
    def split(self, df: pd.DataFrame) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Implementation of the [base splitting method](../reference_splits#splits.balanced_split.BalancedSplit.split).

        Args:
            df (pd.DataFrame): A dataframe of the data. It must contain columns `identity` and `date`.

        Returns:
            List of labels of the training and testing sets.
        """

        df = self.modify_df(df)
        idx_train = list(np.where(df['year'] < self.year)[0])
        if self.test_one_year_only:
            idx_test = list(np.where(df['year'] == self.year)[0])
        else:
            idx_test = list(np.where(df['year'] >= self.year)[0])
        return [(np.array(df.index.values)[idx_train], np.array(df.index.values)[idx_test])]


class TimeCutoffSplitAll(TimeAwareSplit):
    """Sequence of time-cutoff splits TimeCutoffSplit for all possible years.

    Puts all individuals observed before `year` into the training test.
    Puts all individuals observed during `year` into the testing test.
    Implementation of [this paper](https://arxiv.org/abs/2211.10307).
    """

    def __init__(
            self,
            test_one_year_only: bool = True,
            seed: int = 666,
            identity_skip: str = 'unknown',
            ) -> None:
        """Initializes the class.

        Args:
            test_one_year_only (bool, optional): Whether the test set is `df['year'] == year` or `df['year'] >= year`.
            seed (int, optional): Initial seed for the LCG random generator.
            identity_skip (str, optional): Name of the identities to ignore.
        """

        self.test_one_year_only = test_one_year_only
        self.identity_skip = identity_skip
        self.seed = seed
    
    def split(self, df: pd.DataFrame) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Implementation of the [base splitting method](../reference_splits#splits.balanced_split.BalancedSplit.split).

        Args:
            df (pd.DataFrame): A dataframe of the data. It must contain columns `identity` and `date`.

        Returns:
            List of splits. Each split is list of labels of the training and testing sets.
        """

        df = self.modify_df(df)
        years = np.sort(df['year'].unique())[1:]
        splits = []
        for year in years:
            splitter = TimeCutoffSplit(year, self.test_one_year_only)
            for split in splitter.split(df):
                splits.append(split)
        return splits


class RandomProportion():
    # TODO: add documentation
    
    def __init__(self, **kwargs):
        self.splitter = TimeProportionSplit(**kwargs)
    
    def split(self, df):
        splits = []
        for idx_train, idx_test in self.splitter.split(df):
            splits.append(self.splitter.resplit_random(df, idx_train, idx_test))
        return splits