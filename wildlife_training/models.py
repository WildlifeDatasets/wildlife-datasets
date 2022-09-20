import torch
import torch.nn as nn
from timm import create_model

class CategoryCNNClassifier(nn.Module):
    def __init__(self, cnn, classifier):
        super().__init__()
        self.cnn = cnn
        self.classifier = classifier

    def forward(self, batch):
        x, x_category = batch
        x_cnn = self.cnn(x)
        return self.classifier(torch.cat([x_cnn, x_category], dim=1))


def create_model_with_categories(model_name, num_classes, num_categories, **kwargs):
    cnn = create_model(
        model_name = model_name,
        num_classes = 0,
        **kwargs,
        )
    classifier =  nn.Linear(
        in_features = cnn.num_features + num_categories,
        out_features = num_classes,
        bias = True,
        )
    return CategoryCNNClassifier(cnn, classifier)