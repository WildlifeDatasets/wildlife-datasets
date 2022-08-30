import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import faiss

class Evaluation():
    def __init__(self, metrics, method='knn', k=1, batch_size=64, num_workers=0, device='cpu'):
        self.metrics = metrics
        self.k = k
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = device
        if method == 'knn':
            self.predict_func = predict_knn
        elif method == 'classifier':
            self.predict_func = predict_classifier
        else:
            raise ValueError()

    def __call__(self, model, dataset_train, dataset_valid):
        predicted, _ = self.predict_func(
            model,
            dataset_train,
            dataset_valid,
            k=self.k,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            device=self.device,
            )
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


def get_embeddings(embedder, dataset, normalize=True, batch_size=64, num_workers=0, device='cpu'):
    embedder = embedder.eval()
    loader = DataLoader(dataset, shuffle=False, batch_size=batch_size, num_workers=num_workers)

    embeddings = []
    for batch in tqdm(loader):
        img = batch['image'].to(device)
        with torch.no_grad():
            embeddings.append(embedder(img).cpu())
    embeddings = torch.cat(embeddings)

    if normalize:
        embeddings = F.normalize(embeddings)
    return embeddings


def faiss_knn(reference, query, k=1):
    '''
    For each datapoint in query, return K nearest neigbours in reference set.
    '''
    faiss_index = faiss.IndexFlatL2(reference.shape[1])
    faiss_index.add(reference.float().cpu())
    score, index = faiss_index.search(query.float().cpu(), k=k)
    return score, index


def predict_knn(
    embedder,
    dataset_train,
    dataset_valid,
    normalize=True,
    k=1,
    batch_size=64,
    num_workers=0,
    device='cpu',
    **kwargs):
    '''
    Calculates top K predictions using embedder and nearest neighbours in train dataset.

    Example:
        predicted, _ = predict_knn(embedder, dataset_train, dataset_valid)
        actual = dataset_valid.label_map[dataset_valid.label]
        calculate_accuracy(actual, predicted)
    '''

    # Calculate embeddings
    embeddings_train = get_embeddings(
        embedder,
        dataset_train,
        normalize=normalize,
        batch_size=batch_size,
        num_workers=num_workers
    )
    embeddings_valid = get_embeddings(
        embedder,
        dataset_valid,
        normalize=normalize,
        batch_size=batch_size,
        num_workers=num_workers
    )

    # Labels
    if len(dataset_valid.label_map) == 0: # If there are no labels in valid.
        label_name_valid = []
    else:
        label_name_valid = dataset_valid.label_map[dataset_valid.label]
    label_name_train = dataset_train.label_map[dataset_train.label]
    label_train, label_valid, label_map = remap_labels(label_name_train, label_name_valid)

    # Predictions
    score, index = faiss_knn(embeddings_train, embeddings_valid, k)
    predicted = label_map[label_train[index.cpu()]]
    return predicted, score

def predict_classifier(
    classifier,
    dataset_train,
    dataset_valid,
    score_func=None,
    k=1,
    batch_size=64,
    num_workers=0,
    device='cpu',
    **kwargs
    ):
    '''
    Calculates top K predictions using classifier output.

    Example:
        predicted, _ = predict_classifier(classifier, dataset_train, dataset_valid)
        actual = dataset_valid.label_map[dataset_valid.label]
        calculate_accuracy(actual, predicted)
    '''
    classifier = classifier.eval()
    predicted, scores = [], []
    loader = DataLoader(dataset_valid, shuffle=False, batch_size=batch_size, num_workers=num_workers)

    for batch in tqdm(loader):
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
