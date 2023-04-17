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
      lcg (Lcg): Random number generator LCG.
      n (int): Number of samples.
      n_class (int): Number of unique identities.
      y (np.ndarray): List of identities.
      y_counts (np.ndarray): List of sample counts for each unique identity.
      y_unique (np.ndarray): List of unique sorted identities.
    """

    def set_seed(self, seed: int) -> None:
        """Changes the seed of the random number generator.

        Args:
            seed (int): The desired seed.
        """
        
        self.lcg = Lcg(seed)
    
    def split(self, *args, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """Splitting method which needs to be implemented by subclasses.

        It splits the dataframe `df` into labels `idx_train` and `idx_test`.
        The subdataset is obtained by `df.loc[idx_train]` (not `iloc`).

        Returns:
            List of labels of the training and testing sets.
        """

        raise(NotImplementedError('Subclasses should implement this. \n You may want to use ClosedSetSplit instead of BalancedSplit.'))

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

        # Compute how many samples go automatically to the training and testing sets
        y_counts = df['identity'].value_counts()
        n_train = sum([y_counts.loc[y] for y in individual_train])
        n_test = sum([y_counts.loc[y] for y in individual_test])
        
        # Recompute ratio_train and adjust it to proper bounds
        n = len(df)
        ratio_train = self.ratio_train
        if n_train + n_test > 0 and n_train + n_test < n:
            ratio_train = (n*ratio_train - n_train) / (n - n_test - n_train)
        ratio_train = np.clip(ratio_train, 0, 1)
        
        idx_train = np.empty(n, dtype='bool')
        # Make a loop over all individuals
        for individual, y_count in y_counts.iteritems():
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
            idx_train[df['identity'] == individual] = idx_train_class
        return np.array(df.index.values)[idx_train], np.array(df.index.values)[~idx_train]

