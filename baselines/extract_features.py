import numpy as np
import torchvision.transforms as T
from timm import create_model
from wildlife_datasets import datasets
from wildlife_tools.features import DeepFeatures
from wildlife_tools.data import WildlifeDataset

device = 'cuda'
root = '/data/wildlife_datasets/data/WildlifeReID-10k'
d = datasets.WildlifeReID10k(root)

# Extract features by Dinov2
model = create_model("hf-hub:timm/vit_large_patch14_dinov2.lvd142m", pretrained=True).to(device)
extractor = DeepFeatures(model, device=device, batch_size=32)
transform = T.Compose([
    T.Resize(size=518),
    T.CenterCrop(size=[518, 518]),
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])
dataset = WildlifeDataset(metadata=d.df, root=d.root, transform=transform)
features = extractor(dataset)
np.save('features_dino.npy', features)
np.save('features_dino_names.npy', dataset.metadata['path'])
