import os
import numpy as np
import torchvision.transforms as T
from timm import create_model
from wildlife_datasets import datasets
from wildlife_tools.features import DeepFeatures
from transformers import AutoModel

device = 'cuda'
root = '/data/wildlife_datasets/data/WildlifeReID10k'
model_name = 'mega'
if model_name == 'dino':
    img_size = 518
    root_output = 'features_dino'
    model = create_model('hf-hub:timm/vit_large_patch14_dinov2.lvd142m', pretrained=True)
elif model_name == 'mega':
    img_size = 384
    root_output = 'features_mega'
    model = create_model('hf-hub:BVRA/MegaDescriptor-L-384', pretrained=True)
elif model_name == 'miew':
    img_size = 518
    root_output = 'features_miew'
    model = AutoModel.from_pretrained('conservationxlabs/miewid-msv3', trust_remote_code=True)
model = model.to(device)
d = datasets.WildlifeReID10k(root)
os.makedirs(root_output, exist_ok=True)

extractor = DeepFeatures(model, device=device, batch_size=32)
transform = T.Compose([
    T.Resize(size=img_size),
    T.CenterCrop(size=[img_size, img_size]),
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])
for name, df_dataset in d.df.groupby('dataset'):
    file_features = os.path.join(root_output, f'features_{name}.npy')
    file_names = os.path.join(root_output, f'names_{name}.npy')

    if not os.path.exists(file_features):
        dataset = datasets.WildlifeDataset(df=df_dataset, root=d.root, transform=transform, load_label=True)
        features = extractor(dataset)
        if not isinstance(features, np.ndarray):
            features = np.array([f[0] for f in features])    
        np.save(file_features, features)
        np.save(file_names, dataset.metadata['path'])
