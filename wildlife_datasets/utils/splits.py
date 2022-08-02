import numpy as np

class Lcg():
    def __init__(self, seed, iterate=0):
        self.state = seed
        for _ in range(iterate):
            self.random()

    def random(self):
        self.state = (self.state * 1103515245 + 12345) & 0x7FFFFFFF
        return self.state

    def random_permutation(self, n):
        rnd = []
        for _ in range(n):
            self.random()
            rnd.append(self.state)
        return np.argsort(rnd)


class Split():
    # TODO: add unknown to semi?
    # TODO: what to do with csv files?
    def __init__(self, df, seed, keep_unknown=False):
        if keep_unknown:
            self.df = df
        else:
            self.df = df[df['identity'] != 'unknown']
        self.seed = seed

        y = self.df['identity'].to_numpy()                
        _, y_idx, y_counts = np.unique(y, return_index=True, return_counts=True)
        y_unique = np.array([y[index] for index in sorted(y_idx)])
        y_counts = y_counts[np.argsort(y_idx)]

        self.y = y
        self.y_unique = y_unique
        self.y_counts = y_counts
        self.n = len(y)
        self.n_class = len(y_unique)

    def __set_split(self, lcg, ratio_train, individual_train, individual_test, ratio_train_min=0, ratio_train_max=1):
        # check if the intersection of individuals to the train and test sets is empty
        if np.intersect1d(individual_train, individual_test).size > 0:
            raise(Exception('The intersection of individual_train and individual_test must be empty.'))
        
        # how many samples goes automatically to the train and test sets
        n_train = sum(self.y_counts[[k in individual_train for k in self.y_unique]])
        n_test = sum(self.y_counts[[k in individual_test for k in self.y_unique]])
        
        # recompute p and adjust it to proper bounds 
        if n_train + n_test > 0 and n_train + n_test < self.n:
            ratio_train = (self.n*ratio_train - n_train) / (self.n - n_test - n_train)
        ratio_train = np.minimum(np.maximum(ratio_train, ratio_train_min), ratio_train_max)
        
        idx_train = np.empty(self.n, dtype='bool')
        for individual, y_count in zip(self.y_unique, self.y_counts):            
            # check if the class does not go fully to the train or test set
            if individual in individual_train:
                idx_train_class = np.ones(y_count, dtype='bool')
            elif individual in individual_test:
                idx_train_class = np.zeros(y_count, dtype='bool')
            else:
                idx_train_class = np.zeros(y_count, dtype='bool')

                # number of samples to the train set
                n_train = np.round(ratio_train * y_count).astype(int)
                if n_train == y_count and n_train > 1:
                    n_train -= 1
                if n_train == 0:
                    n_train = 1

                # create indices and randomly permute them                
                idx_permutation = lcg.random_permutation(y_count)
                idx_train_class[:n_train] = True                
                idx_train_class = idx_train_class[idx_permutation]

            idx_train[self.y == individual] = idx_train_class
        return np.array(self.df.index.values)[idx_train], np.array(self.df.index.values)[~idx_train]

    def closed_set_split(self, ratio_train):
        lcg = Lcg(self.seed, 2)
        individual_train = np.array([], dtype=object)
        individual_test = np.array([], dtype=object)
        return self.__set_split(lcg, ratio_train, individual_train, individual_test)

    def open_set_split(self, ratio_train, ratio_class_test):
        lcg = Lcg(self.seed, 3)
        idx_permuted = lcg.random_permutation(self.n_class)
        y_unique = self.y_unique[idx_permuted]
        y_counts = self.y_counts[idx_permuted]        
        i_end = np.where(np.cumsum(y_counts) >= np.round(ratio_class_test * self.n).astype(int))[0][0]
        individual_train = np.array([], dtype=object)
        individual_test = np.array(y_unique[:i_end])
        return self.__set_split(lcg, ratio_train, individual_train, individual_test)

    def disjoint_set_split(self, ratio_class_test):
        lcg = Lcg(self.seed, 4)
        idx_permuted = lcg.random_permutation(self.n_class)
        y_unique = self.y_unique[idx_permuted]
        y_counts = self.y_counts[idx_permuted]        
        i_end = np.where(np.cumsum(y_counts) >= np.round(ratio_class_test * self.n).astype(int))[0][0]
        individual_train = np.array(y_unique[i_end:])
        individual_test = np.array(y_unique[:i_end])
        return self.__set_split(lcg, [], individual_train, individual_test)
