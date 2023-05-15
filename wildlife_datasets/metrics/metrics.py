import numpy as np
import sklearn.metrics as skm

# TODO: add documentation
# TODO: check all code

def unify_types(y_true, y_pred, new_class):
    y_all = list(set(y_true).union(set(y_pred)) - set([new_class]))        
    y_types = set([type(y) for y in y_all])
    if len(y_types) > 1:
        raise(Exception('Labels have mixed types. Convert all to int or str.'))
    if str in y_types and isinstance(new_class, int):
        encoder = {new_class: new_class}
        for i, y in enumerate(y_all):
            encoder[y] = new_class + 1 + i
        y_true = [encoder[y] for y in y_true]
        y_pred = [encoder[y] for y in y_pred]
    if int in y_types and isinstance(new_class, str):
        encoder = {new_class: np.min(y_all) - 1}
        for i, y in enumerate(y_all):
            encoder[y] = y
        y_true = [encoder[y] for y in y_true]
        y_pred = [encoder[y] for y in y_pred]
        new_class = encoder[new_class]
    return y_true, y_pred, new_class


def accuracy(
        y_true,
        y_pred,
        new_class=None,
    ):
    y_true, y_pred, new_class = unify_types(y_true, y_pred, new_class)
    return np.mean(np.array(y_pred) == np.array(y_true))

def balanced_accuracy(
        y_true,
        y_pred,
        new_class=None,
    ):
    y_true, y_pred, new_class = unify_types(y_true, y_pred, new_class)    
    C = skm.confusion_matrix(y_true, y_pred)
    with np.errstate(divide="ignore", invalid="ignore"):
        per_class = np.diag(C) / C.sum(axis=1)
    return np.mean(per_class[~np.isnan(per_class)])

def class_average_accuracy(
        y_true,
        y_pred,
        new_class=None,
    ):
    y_true, y_pred, new_class = unify_types(y_true, y_pred, new_class)
    C = skm.multilabel_confusion_matrix(y_true, y_pred)
    return np.mean([C_i[0,0]+C_i[1,1] for C_i in C]) / np.sum(C[0])

def precision(
        y_true,
        y_pred,
        new_class=None,
    ):
    y_true, y_pred, new_class = unify_types(y_true, y_pred, new_class)
    return skm.precision_score(y_true, y_pred, average='macro')

def recall(
        y_true,
        y_pred,
        new_class=None,
        ignore_empty=False
    ):
    y_true, y_pred, new_class = unify_types(y_true, y_pred, new_class)
    if ignore_empty:
        C = skm.multilabel_confusion_matrix(y_true, y_pred)
        return np.mean([C_i[1,1]/(C_i[1,0]+C_i[1,1]) for C_i in C if C_i[1,0]+C_i[1,1] > 0])
    else:
        return skm.recall_score(y_true, y_pred, average='macro')

def f1(
        y_true,
        y_pred,
        new_class=None,
    ):
    y_true, y_pred, new_class = unify_types(y_true, y_pred, new_class)
    return skm.f1_score(y_true, y_pred, average='macro')

def accuracy_known_samples(
        y_true,
        y_pred,
        new_class,
    ):
    y_true, y_pred, new_class = unify_types(y_true, y_pred, new_class)
    y_true = np.array(y_true)    
    y_pred = np.array(y_pred)
    known = y_true != new_class
    if sum(known) > 0:
        return np.mean(y_true[known] == y_pred[known])
    else:
        return np.nan

def accuracy_unknown_samples(
        y_true,
        y_pred,
        new_class,
    ):
    y_true, y_pred, new_class = unify_types(y_true, y_pred, new_class)
    y_true = np.array(y_true)    
    y_pred = np.array(y_pred)
    unknown = y_true == new_class
    if sum(unknown) > 0:
        return np.mean(y_true[unknown] == y_pred[unknown])
    else:
        return np.nan
    
def normalized_accuracy(
        y_true,
        y_pred,
        new_class,
        mu
    ):
    aks = accuracy_known_samples(y_true, y_pred, new_class)
    aus = accuracy_unknown_samples(y_true, y_pred, new_class)
    return mu*aks + (1-mu)*aus

def average_precision(
        y_true,
        y_pred
    ):
    unify_types([y_true], y_pred, None)
    a = np.array(y_pred) == y_true
    b = np.linspace(1, 0, len(y_pred))
    if sum(a) == 0:
        return 0
    else:
        return skm.average_precision_score(a, b)

def mean_average_precision(
        y_true,
        y_pred
    ):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean([average_precision(y_t, y_p) for y_t, y_p in zip(y_true, y_pred)])

def auc_roc_new_class(
        y_true,
        y_score,
        new_class,
    ):
    # TODO: check if it is correct
    # TODO: low y_score means high chance of new individual
    y_true, _, new_class = unify_types(y_true, [], new_class)    
    a = np.array(y_true) == new_class
    b = -np.array(y_score)
    return skm.roc_auc_score(a, b)
