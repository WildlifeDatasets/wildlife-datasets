import os
import sys
sys.path.append(os.path.dirname(__file__))
from baseline_arcface import *
config['name'] = os.path.splitext(os.path.basename(__file__))[0]



# Prepare trainer
def create_trainer(dataset):
    embedder = create_model(
        model_name = 'efficientnet_b0',
        pretrained = True,
        num_classes = config['embedding_size'],
        )

    loss_func = losses.SubCenterArcFaceLoss(
        num_classes = dataset.num_classes,
        embedding_size = config['embedding_size'],
        margin = 28.6,
        scale = 64,
        sub_centers = 5,
        )

    optimizers = {
        'embedder': torch.optim.Adam(
            params=embedder.parameters(),
            lr=1e-3
            ),
        'loss': torch.optim.Adam(
            params=loss_func.parameters(),
            lr=1e-3
            ),
        }

    trainer = EmbeddingTrainer(
        embedder = embedder,
        loss_func = loss_func,
        optimizers = optimizers,
        evaluation = evaluation,
        device = config['device'],
    )
    return trainer