import numpy as np
from typing import List

# TODO: add documentation
class Lcg():
    '''        
    Simple Linear congruential generator for generating random numbers.
    Copied from https://stackoverflow.com/questions/18634079/glibc-rand-function-implementation        
    It is machine-, distribution- and package version-independent.
    It has some drawbacks (check the link above) but perfectly sufficient for our application.
    '''
    def __init__(self, seed: int, iterate: int=0) -> None:
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

    def random_shuffle(self, x: np.ndarray) -> np.ndarray:
        return x[self.random_permutation(len(x))]
