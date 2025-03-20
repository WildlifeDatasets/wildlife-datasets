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
        df = df[df[self.col_label] != self.identity_skip]

        # Removes entries without dates
        df = df[~df['date'].isnull()]
        
        # Convert date to datetime format (from possibly strings) and drop hours
        df['date'] = pd.to_datetime(df['date']).apply(lambda x: x.date())
        df['year'] = df['date'].apply(lambda x: x.year).to_numpy()            
        return df


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
            **kwargs
            ):
        """Initializes the class.

        Args:
            ratio (float, optional): The fraction of dates going to the training set.
            **kwargs (type, optional): See kwargs `seed`, `identity_skip` and `col_label` of the parent class.
        """

        self.ratio = ratio
        super().__init__(**kwargs)

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
        for _, df_name in df.groupby(self.col_label):            
            dates = df_name.groupby('date')
            n_dates = len(dates)
            if n_dates > 1:
                # Loop over all dates; y is a tuple (date, df with unique date and identity)
                for i, (_, df_date) in enumerate(dates):
                    # Add half dates to the training and half to the testing set
                    if i < int(np.round(self.ratio*n_dates)):
                        idx_train += list(df_date.index)
                    else:
                        idx_test += list(df_date.index)
            else:
                idx_train += list(df_name.index)
        return [(np.array(idx_train), np.array(idx_test))]


class TimeProportionOpenSetSplit(TimeAwareSplit):
    """Time-proportion open set splitting method into training and testing sets.

    First, it pust some individuals into the training set only.
    Then it is the TimeProportionSplit.
    """

    def __init__(
            self,
            ratio_train: float,
            ratio_class_test: float = None,
            n_class_test: int = None,
            **kwargs
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
        super().__init__(**kwargs)

    def split(self, df: pd.DataFrame) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Implementation of the [base splitting method](../reference_splits#splits.balanced_split.BalancedSplit.split).

        Args:
            df (pd.DataFrame): A dataframe of the data. It must contain columns `identity` and `date`.

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

        # Compute how many samples go automatically to the training and testing sets
        y_counts = df[self.col_label].value_counts()
        n_train = sum([y_counts.loc[y] for y in individual_train])
        n_test = sum([y_counts.loc[y] for y in individual_test])
        
        # Recompute ratio_train and adjust it to proper bounds
        ratio_train = self.ratio_train
        if n_train + n_test > 0 and n_train + n_test < n:
            ratio_train = (n*ratio_train - n_train) / (n - n_test - n_train)
        ratio_train = np.clip(ratio_train, 0, 1)

        idx_train = []
        idx_test = []
        # Loop over all identities; x is a tuple (identity, df with unique identity)
        for name, df_name in df.groupby(self.col_label):
            if name in individual_train and name in individual_test:
                # Check if the class does not belong to both sets
                raise(Exception('Individual cannot be both in individual_train and individual_test.'))
            elif name in individual_train:
                # Check if the class does not belong to the training set
                idx_train += list(df_name.index)
            elif name in individual_test:
                # Check if the class does not belong to the testing set
                idx_test += list(df_name.index)
            else:
                dates = df_name.groupby('date')
                n_dates = len(dates)
                if n_dates > 1:
                    # Loop over all dates; y is a tuple (date, df with unique date and identity)
                    for i, (_, df_date) in enumerate(dates):
                        # Add half dates to the training and half to the testing set
                        if i < int(np.round(ratio_train*n_dates)):
                            idx_train += list(df_date.index)
                        else:
                            idx_test += list(df_date.index)
                else:
                    idx_train += list(df_name.index)
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
            **kwargs
            ) -> None:
        """Initializes the class.

        Args:
            year (int): Splitting year.
            test_one_year_only (bool, optional): Whether the test set is `df['year'] == year` or `df['year'] >= year`.
            **kwargs (type, optional): See kwargs `seed`, `identity_skip` and `col_label` of the parent class.
        """

        self.year = year
        self.test_one_year_only = test_one_year_only
        super().__init__(**kwargs)
    
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
            **kwargs
            ) -> None:
        """Initializes the class.

        Args:
            test_one_year_only (bool, optional): Whether the test set is `df['year'] == year` or `df['year'] >= year`.
            **kwargs (type, optional): See kwargs `seed`, `identity_skip` and `col_label` of the parent class.
        """

        self.test_one_year_only = test_one_year_only
        super().__init__(**kwargs)
    
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
            splitter = TimeCutoffSplit(
                year,
                test_one_year_only=self.test_one_year_only,
                seed=self.seed,
                identity_skip=self.identity_skip,
                col_label=self.col_label
                )
            for split in splitter.split(df):
                splits.append(split)
        return splits


class RandomProportion():
    """Wrapper for resplits of TimeProportionSplit.
    """

    def __init__(self, **kwargs):
        self.splitter = TimeProportionSplit(**kwargs)
    
    def split(self, df):
        splits = []
        for idx_train, idx_test in self.splitter.split(df):
            splits.append(self.splitter.resplit_random(df, idx_train, idx_test))
        return splits
    
    def set_col_label(self, *args, **kwargs):
        self.splitter.set_col_label(*args, **kwargs)
