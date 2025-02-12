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
            ratio (float, optional): The fraction of dates going to the testing set.
            seed (int, optional): Initial seed for the LCG random generator.
            identity_skip (str, optional): Name of the identities to ignore.
            col_label (str, optional): Column name containing individual animal names (labels).            
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
            seed (int, optional): Initial seed for the LCG random generator.            
            identity_skip (str, optional): Name of the identities to ignore.
            col_label (str, optional): Column name containing individual animal names (labels).
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
            seed (int, optional): Initial seed for the LCG random generator.
            identity_skip (str, optional): Name of the identities to ignore.
            col_label (str, optional): Column name containing individual animal names (labels).
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
