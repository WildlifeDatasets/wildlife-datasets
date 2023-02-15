# How to use splitting functions

The crucial part of machine learning is training a method on a training set and evaluating it on a separate testing set. The creation of training and testing set may significantly influence the results. The splits may be performed based on identities or on time. Due to our representation of the [random number generator](../reference_splits#lcg), the splits should be both machine- and system-independent. We follow the presentation from [this paper](https://arxiv.org/abs/2211.10307).

We assume that we have already [downloaded](../tutorial_datasets#downloading-datasets) the dataset. Then we load the dataset and the dataframe.

    from wildlife_datasets import datasets, splits

    dataset = datasets.MacaqueFaces('data/MacaqueFaces')
    df = dataset.df
    seed = 666


## Splits based on identities

Splits on identities perform the split for each individual separetely. All these splits are random.

### Closed-set split

The most common split is the closed-set split, where each individual has samples in both the training and testing sets.

    splitter = splits.ClosedSetSplit(df, seed)
    idx_train, idx_test = splitter.split(0.85)

This code generates a split, where the training set contains approximately 85% of all samples. However, since the split is done separately for each individual, the training set actually contains 85.13% of all samples. The outputs of the spliller are labels, not indices, and, therefore, we access the training and testing sets by

    df_train, df_test = df.loc[idx_train], df.loc[idx_test]

### Open-set split

In the open-set split, there are some individuals with all their samples only in the testing set. For the remaining individuals, the closed-set split is performed.

    splitter = splits.OpenSetSplit(df, seed)
    idx_train, idx_test = splitter.split(0.5, 0.1)

This code generates a split, where approximately 10% of samples are put directly into the testing set. It also specifies that the training set should contain 50% of all samples. As in the previous (and all following) cases, the numbers are only approximate with the actual ratios being 8.92% and 49.92%. The other possibility to create this split is to prescribe the number of individuals (instead of ratio of samples) which go directly into the testing set.

    splitter = splits.OpenSetSplit(df, seed)
    idx_train, idx_test = splitter.split(0.5, n_class_test=5)

### Disjoint-set split

For the disjoint-set split, each individual has all samples either in the training or testing set but never in both. Similarly as in the open-set split, we can create the split either by

    splitter = splits.DisjointSetSplit(df, seed)
    idx_train, idx_test = splitter.split(0.5)

or

    splitter = splits.DisjointSetSplit(df, seed)
    idx_train, idx_test = splitter.split(n_class_test=10)

The first method put approximately 50% of the samples to the testing set, while the second method puts 10 classes to the testing set.


## Splits based on time

Splits based on time create some cutoff time and put everything before the cutoff time into the training set and everything after the cutoff time into the training set. Therefore, this splits are not random but deterministic. These splits also ignore all samples without timestamps.

### Time-proportion split

Time-proportion split counts on how many days was each individual observed. Then it puts all samples corresponding to the first half of the observation days to the training set and all remaining to the testing set. It ignores all individuals observed only on one day. Since all individuals are both in the training and testing set, it leads to the closed-set split.

    splitter = TimeProportionSplit(df, seed)
    idx_train, idx_test = splitter.split()

Even though the split is non-random, it still required the seed because of the [random resplit](#random-resplit).

### Time-cutoff split

While the time-proportion day selected a different cutoff day for each individual, the time-cutoff split creates one cutoff year for the whole dataset. All samples taken before the cutoff year go the training set, while all samples taken during the cutoff year go to the testing set. Since some individuals may be present only in the testing set, this split is usually an open-set split.

    splitter = TimeCutoffSplit(df, seed)
    idx_train, idx_test = splitter.split(2015)

It is also possible to place all samples taken during or after the cutoff year to the testing set by

    splitter = TimeCutoffSplit(df, seed)
    idx_train, idx_test = splitter.split(2015, test_only_year=False)

It is also possible to create all possible time-cutoff splits for different years by

    splitter = TimeCutoffSplit(df, seed)
    splits, cutoff_years = splitter.splits_all()

### Random resplit

Since the splits based on time are not random, it is also possible to create similar random splits by 

    idx_train, idx_test = splitter.resplit_random(idx_train, idx_test)

For each individual the number of samples in the training set will be the same for the original and new splits. Since the new splits are random, they do not utilize the time at all.
