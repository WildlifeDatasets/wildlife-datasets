from ..splits import Lcg
import numpy as np
import sklearn.metrics as skm
import copy

# TODO: add documentation
# TODO: check all code

def unify_types(y_true, y_pred, unknown_class):
    y_all = list(set(y_true).union(set(y_pred)) - set([unknown_class]))        
    y_types = set([type(y) for y in y_all])
    if len(y_types) > 1:
        raise(Exception('Mixed labels found.'))
    if str in y_types and isinstance(unknown_class, int):
        encoder = {unknown_class: unknown_class}
        for i, y in enumerate(y_all):
            encoder[y] = unknown_class + 1 + i
        y_true = [encoder[y] for y in y_true]
        y_pred = [encoder[y] for y in y_pred]
    if int in y_types and isinstance(unknown_class, str):
        encoder = {unknown_class: np.min(y_all) - 1}
        for i, y in enumerate(y_all):
            encoder[y] = y
        y_true = [encoder[y] for y in y_true]
        y_pred = [encoder[y] for y in y_pred]
        unknown_class = encoder[unknown_class]
    return y_true, y_pred, unknown_class


def accuracy(
        y_true,
        y_pred,
        unknown_class=None,
    ):
    y_true, y_pred, unknown_class = unify_types(y_true, y_pred, unknown_class)
    return np.mean(np.array(y_pred) == np.array(y_true))

def balanced_accuracy(
        y_true,
        y_pred,
        unknown_class=None,
    ):
    y_true, y_pred, unknown_class = unify_types(y_true, y_pred, unknown_class)    
    C = skm.confusion_matrix(y_true, y_pred)
    with np.errstate(divide="ignore", invalid="ignore"):
        per_class = np.diag(C) / C.sum(axis=1)
    return np.mean(per_class[~np.isnan(per_class)])

def class_average_accuracy(
        y_true,
        y_pred,
        unknown_class=None,
    ):
    y_true, y_pred, unknown_class = unify_types(y_true, y_pred, unknown_class)
    C = skm.multilabel_confusion_matrix(y_true, y_pred)
    return np.mean([C_i[0,0]+C_i[1,1] for C_i in C]) / np.sum(C[0])

def precision(
        y_true,
        y_pred,
        unknown_class=None,
    ):
    y_true, y_pred, unknown_class = unify_types(y_true, y_pred, unknown_class)
    return skm.precision_score(y_true, y_pred, average='macro')

def recall(
        y_true,
        y_pred,
        unknown_class=None,
        ignore_empty=False
    ):
    y_true, y_pred, unknown_class = unify_types(y_true, y_pred, unknown_class)
    if ignore_empty:
        C = skm.multilabel_confusion_matrix(y_true, y_pred)
        return np.mean([C_i[1,1]/(C_i[1,0]+C_i[1,1]) for C_i in C if C_i[1,0]+C_i[1,1] > 0])
    else:
        return skm.recall_score(y_true, y_pred, average='macro')

def f1(
        y_true,
        y_pred,
        unknown_class=None,
    ):
    y_true, y_pred, unknown_class = unify_types(y_true, y_pred, unknown_class)
    return skm.f1_score(y_true, y_pred, average='macro')
    
def normalized_accuracy(
        y_true,
        y_pred,
        unknown_class,
        mu
    ):
    y_true, y_pred, unknown_class = unify_types(y_true, y_pred, unknown_class)
    y_true = np.array(y_true)    
    y_pred = np.array(y_pred)
    known = y_true == unknown_class
    if sum(known) == 0:
        raise(Exception('Some true labels must be of the unknown class.'))
    
    aks = np.mean(y_true[known] == y_pred[known])
    aus = np.mean(y_true[~known] == y_pred[~known])
    return mu*aks + (1-mu)*aus