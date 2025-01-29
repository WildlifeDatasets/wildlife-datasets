import os
import numpy as np
import torchvision.transforms as T
from timm import create_model
from wildlife_datasets import datasets
from wildlife_tools.features import DeepFeatures

device = 'cuda'
root = '/data/wildlife_datasets/data/WildlifeReID10k'
root_output = 'features'
d = datasets.WildlifeReID10k(root)
os.makedirs(root_output, exist_ok=True)

# Extract features by Dinov2
model = create_model("hf-hub:timm/vit_large_patch14_dinov2.lvd142m", pretrained=True).to(device)
extractor = DeepFeatures(model, device=device, batch_size=32)
transform = T.Compose([
    T.Resize(size=518),
    T.CenterCrop(size=[518, 518]),
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])
for name, df_dataset in d.df.groupby('dataset'):
    dataset = datasets.WildlifeDataset(df=df_dataset, root=d.root, transform=transform, load_label=True)
    features = extractor(dataset)
    if not isinstance(features, np.ndarray):
        features = np.array([f[0] for f in features])    
    np.save(os.path.join(root_output, f'features_{name}.npy'), features)
    np.save(os.path.join(root_output, f'names_{name}.npy'), dataset.metadata['path'])
