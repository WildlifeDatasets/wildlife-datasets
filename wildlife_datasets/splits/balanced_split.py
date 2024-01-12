import numpy as np
import pandas as pd
from typing import List, Tuple
from .lcg import Lcg


class BalancedSplit():
    # TODO: change docs
    """Base class for splitting datasets into training and testing sets.

    Implements methods from [this paper](https://arxiv.org/abs/2211.10307).
    Its subclasses need to implement the `split` method.
    It should perform balanced splits separately for all classes.
    Its children are `IdentitySplit` and `TimeAwareSplit`.
    `IdentitySplit` has children `ClosedSetSplit`, `OpenSetSplit` and `DisjointSetSplit`.
    `TimeAwareSplit` has children `TimeProportionSplit` and `TimeCutoffSplit`.

    Attributes:
      df (pd.DataFrame): A dataframe of the data. It must contain columns
        `identity` for all splits and `date` for time-aware splits.
      seed (int): Initial seed for the random number generator LCG.
      n (int): Number of samples.
      n_class (int): Number of unique identities.
      y (np.ndarray): List of identities.
      y_counts (np.ndarray): List of sample counts for each unique identity.
      y_unique (np.ndarray): List of unique sorted identities.
    """

    def initialize_lcg(self) -> Lcg:
        """Returns the random number generator.

        Returns:
            The random number generator.
        """
        
        return Lcg(self.seed)
    
    def split(self, *args, **kwargs) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Splitting method which needs to be implemented by subclasses.

        It splits the dataframe `df` into labels `idx_train` and `idx_test`.
        The subdataset is obtained by `df.loc[idx_train]` (not `iloc`).

        Returns:
            List of splits. Each split is list of labels of the training and testing sets.
        """

        raise(NotImplementedError('Subclasses should implement this. \n You may want to use ClosedSetSplit instead of BalancedSplit.'))


