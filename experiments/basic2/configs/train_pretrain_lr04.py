import os
import sys
sys.path.append(os.path.dirname(__file__))
from baseline import *
config['name'] = os.path.splitext(os.path.basename(__file__))[0]
config['epochs'] = 200

def create_trainer(dataset):
    model = create_model(
        model_name = 'efficientnet_b0',
        pretrained = True,
        num_classes = dataset.num_classes,
        )

    optimizer = torch.optim.Adam(
        params = model.parameters(),
        lr = 1e-4
        )

    trainer = BasicTrainer(
        model = model,
        evaluation = evaluation,
        optimizer = optimizer,
        device = config['device'],
    )
    return trainer