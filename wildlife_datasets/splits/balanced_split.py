import numpy as np
import pandas as pd
from typing import List, Tuple
from .lcg import Lcg

# TODO: add documentation
class BalancedSplit():
    """Base class for splitting datasets into training and testing sets.

    Its subclasses need to implement the split() method.
    It should perform balanced splits separately for all classes.
    Its children are ClosedSetSplit, OpenSetSplit, DisjointSetSplit and TimeAwareSplit.
    TimeAwareSplit furher has children TimeProportionSplit and TimeCutoffSplit.

    Attributes:
      df (pd.DataFrame): A dataframe of the data. It must contain columns
        `identity` for all splits and `date` for time-aware splits.
      # TODO: finish
    """

    def __init__(self, df: pd.DataFrame, seed: int, identity_skip: str = 'unknown') -> None:
        """Initializes the class.

        Args:
            df (pd.DataFrame): A dataframe of the data. It must contain columns
                `identity` for all splits and `date` for time-aware splits.
            seed (int): Initial seed for the LCG random generator.
            identity_skip (str, optional): Name of the identities to ignore.
        """

        # Potentially remove the unknown identities
        self.df = df.copy()
        self.df = self.df[self.df['identity'] != identity_skip]
        # Initialize the random number generator
        self.change_seed(seed)

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

