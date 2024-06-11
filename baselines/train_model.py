from itertools import chain
import torch
import timm
import torchvision.transforms as T
from torch.optim import SGD
from wildlife_datasets import datasets
from wildlife_tools.data import WildlifeDataset, SplitMetadata
from wildlife_tools.features import DeepFeatures
from wildlife_tools.train import ArcFaceLoss, BasicTrainer

# Set data for similarity-aware and random splits
data = [
    ('split', 'results/MD-L-384_similarity', 'data/MD-L-384_similarity.pth'),
    ('split_random', 'results/MD-L-384_random', 'data/MD-L-384_similarity.pth')
]

for split_col, folder, file_name_features in data:
    # Dataset configuration
    root = '/data/wildlife_datasets/data/WildlifeReID10k'
    d = datasets.WildlifeReID10k(root)
    transform = T.Compose([
        T.Resize(size=(384, 384)),
        T.RandAugment(num_ops=2, magnitude=20),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    dataset = WildlifeDataset(
        metadata = d.df, 
        root = d.root,
        split = SplitMetadata(split_col, 'train'),
        transform=transform
    )

    # Backbone and loss configuration
    backbone = timm.create_model('swin_large_patch4_window12_384', num_classes=0, pretrained=True)
    with torch.no_grad():
        dummy_input = torch.randn(1, 3, 384, 384)
        embedding_size = backbone(dummy_input).shape[1]
    objective = ArcFaceLoss(num_classes=dataset.num_classes, embedding_size=embedding_size, margin=0.5, scale=64)

    # Optimizer and scheduler configuration
    params = chain(backbone.parameters(), objective.parameters())
    optimizer = SGD(params=params, lr=0.001, momentum=0.9)
    min_lr = optimizer.defaults.get("lr") * 1e-3
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=min_lr)

    # Setup training
    trainer = BasicTrainer(
        dataset=dataset,
        model=backbone,
        objective=objective,
        optimizer=optimizer,
        scheduler=scheduler,
        batch_size=16,
        accumulation_steps=8,
        num_workers=2,
        epochs=100,
        device='cuda',
    )

    # Train
    trainer.train()
    trainer.save(folder)

    # Prepare for feature exctraction
    path = f'{folder}/checkpoint.pth'
    transform = T.Compose([
        T.Resize(size=(384, 384)),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    dataset = WildlifeDataset(metadata=d.df, root=d.root, transform=transform)
    model = timm.create_model("swin_large_patch4_window12_384", pretrained=True, num_classes=0)
    model.load_state_dict(torch.load(path)['model'])
    model = model.to('cuda')

    # Extract features
    extractor = DeepFeatures(model, device='cuda', batch_size=64)
    features=extractor(dataset)
    torch.save(features, file_name_features)
