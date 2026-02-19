
import numpy as np


class Lcg():
    """Linear congruential generator for generating random numbers.
    
    Copied from [StackOverflow](https://stackoverflow.com/questions/18634079/glibc-rand-function-implementation).
    It is machine-, distribution- and package version-independent.
    It has some drawbacks (check the link above) but perfectly sufficient for our application.

    Attributes:
      state (int): Random state of the LCG.
    """
    
    def __init__(self, seed: int, iterate: int=0) -> None:
        """Initialization function for LCG.

        Args:
            seed (int): Initial random seed.
            iterate (int, optional): Number of initial random iterations.
        """
        self.state = seed
        for _ in range(iterate):
            self.random()

    def random(self) -> int:
        """Generates a new random integer from the current state.

        Returns:
            New random integer.
        """

        self.state = (self.state * 1103515245 + 12345) & 0x7FFFFFFF
        return self.state

    def random_permutation(self, n: int) -> np.ndarray:
        """Generates a random permutation of `range(n)`.

        Args:
            n (int): Length of the sequence to be permuted.

        Returns:
            Permuted sequence.
        """
        
        rnd = []
        for _ in range(n):
            self.random()
            rnd.append(self.state)
        return np.argsort(rnd)

    def random_shuffle(self, x: np.ndarray) -> np.ndarray:
        """Generates a random shuffle of `x`.

        Args:
            x (np.ndarray): Array to be permuted.

        Returns:
            Shuffled array.
        """

        return np.array(x)[self.random_permutation(len(x))]
