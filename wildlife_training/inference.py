
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pytorch_metric_learning.utils.inference import FaissKNN
from pytorch_metric_learning import testers


class Evaluation():
    def __init__(self, metrics, method='knn', device='cpu', k=1):
        self.metrics = metrics
        self.device = device
        self.k = k
        if method == 'knn':
            self.predict_func = predict_knn
        elif method == 'classifier':
            self.predict_func = predict_classifier
        else:
            raise ValueError()

    def __call__(self, model, dataset_train, dataset_valid):
        predicted, _ = self.predict_func(model, dataset_train, dataset_valid, k=self.k, device=self.device)
        actual = dataset_valid.label_map[dataset_valid.label]

        output = {}
        for name, metric in self.metrics.items():
            output[name] = metric(actual, predicted)
        return output


def remap_labels(label_train, label_valid):
    '''
    Map two potentially disjoint label sets to the same integer labels.
    
    Example: 
        remap_labels(['x', 'y', 'z'], ['x', 'x', 'w'])
        >>> (array([1, 3, 2]), array([1, 1, 0]), array(['w', 'x', 'z', 'y'], dtype=object))

    '''
    label, label_all_map = pd.factorize(list(set.union(set(label_train), set(label_valid))))
    label_all_dict = {value: i for i, value in enumerate(label_all_map)}

    label_train = list(map(lambda x: label_all_dict[x], label_train))
    label_valid = list(map(lambda x: label_all_dict[x], label_valid))
    return np.array(label_train), np.array(label_valid), np.array(label_all_map)


def predict_knn(embedder, dataset_train, dataset_valid, k=5, **kwargs):
    '''
    Calculates top K predictions using embedder and nearest neighbours in train dataset.

    Example:
        predicted, _ = predict_knn(embedder, dataset_train, dataset_valid)
        actual = dataset_valid.label_map[dataset_valid.label]
        calculate_accuracy(actual, predicted)
    '''
    # Embeddings
    embedder = embedder.eval()
    tester = testers.BaseTester(data_and_label_getter=lambda x: (x['image'], x['label']))
    embeddings_train, _ = tester.get_all_embeddings(dataset_train, embedder)
    embeddings_valid, _ = tester.get_all_embeddings(dataset_valid, embedder)

    # Labels
    if len(dataset_valid.label_map) == 0: # If there are no labels in valid.
        label_name_valid = []
    else:
        label_name_valid = dataset_valid.label_map[dataset_valid.label]
    label_name_train = dataset_train.label_map[dataset_train.label]
    label_train, label_valid, label_map = remap_labels(label_name_train, label_name_valid)

    # Predictions
    knn_function = FaissKNN()
    score, index = knn_function(embeddings_valid, k, embeddings_train)
    predicted = label_map[label_train[index.cpu()]]
    return predicted, score


def predict_classifier(classifier, dataset_train, dataset_valid, k=5, batch_size=64, score_func=None, device='cpu', **kwargs):
    '''
    Calculates top K predictions using classifier output.

    Example:
        predicted, _ = predict_classifier(classifier, dataset_train, dataset_valid)
        actual = dataset_valid.label_map[dataset_valid.label]
        calculate_accuracy(actual, predicted)
    '''
    classifier = classifier.eval()
    predicted, scores = [], []
    loader = DataLoader(dataset_valid, shuffle=False, batch_size=batch_size)

    for batch in loader:
        img = batch['image'].to(device)
        with torch.no_grad():
            output = classifier(img).cpu()

        # Calculate score
        if score_func is None:
            score = output
        elif score_func == 'log_softmax':
            score = torch.exp(F.log_softmax(output, dim=1))
        elif score_func == 'softmax':
            score = F.log_softmax(output, dim=1)
        else:
            raise ValueError(f'Invalid score function: {score_func}')

        score, index = score.topk(k, dim=1)
        predicted.append(dataset_train.label_map[index.numpy()])
        scores.append(score.numpy())

    predicted = np.concatenate(predicted)
    scores = np.concatenate(scores)
    return predicted, scores
