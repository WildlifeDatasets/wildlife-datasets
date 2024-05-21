import os
import numpy as np
import pandas as pd
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm
from wildlife_datasets import datasets
from wildlife_tools.data.dataset import WildlifeDataset


def resize_dataset(
        dataset_factory,
        new_root,
        idx=None,
        size=None,
        img_load="bbox",
        copy_files=True
        ):
    
    # Create dataset loader
    if idx is None:
        idx = np.where(dataset_factory.df['identity'] != 'unknown')[0]
    if size is not None:
        transform = T.Resize(size=size)
    else:
        transform = None
    dataset = WildlifeDataset(dataset_factory.df, dataset_factory.root,
        transform=transform, img_load=img_load)

    df_new = []
    for i in tqdm(idx, mininterval=1, ncols=100):
        if dataset.metadata.iloc[i]['identity'] != 'unknown':            
            # Make image path unique across datasets
            row = dataset.metadata.iloc[i]
            base, ext = os.path.splitext(row["path"])
            img_path = base + "_" + str(row["image_id"]) + ext
            img_path = img_path.replace('\\\\', '/')
            img_path = img_path.replace('\\', '/')

            # Save image to new root with unique image path
            if copy_files:
                image, _ = dataset[i]
                full_img_path = os.path.join(new_root, img_path)
                if not os.path.exists(os.path.dirname(full_img_path)):
                    os.makedirs(os.path.dirname(full_img_path))
                image.save(full_img_path)

            # Extract species
            if 'species' in row:
                species = row['species']
            else:
                species = dataset_factory.metadata['animals']
                if len(species) != 1:
                    raise Exception('There should be only one species')
                species = list(species)[0]

            # Update dataframe
            df_new.append({
                'image_id': row['image_id'],
                'identity': row['identity'],
                'path': img_path,
                'species': species,
                'date': row.get('date', np.nan),
                'orientation': row.get('orientation', np.nan),
            })
    return pd.DataFrame(df_new)

def prepare_aau_zebrafish(root, new_root="data/AAUZebraFish", **kwargs):
    dataset_factory = datasets.AAUZebraFish(root)
    return resize_dataset(dataset_factory, new_root, img_load="bbox", **kwargs)

def prepare_aerial_cattle_2017(root, new_root="data/AerialCattle2017", **kwargs):
    dataset_factory = datasets.AerialCattle2017(root)
    # Take only every tenth frame in videos
    idx = np.array([], dtype=int)
    for _, df_red in dataset_factory.df.groupby(['identity', 'video']):
        idx = np.hstack((idx, df_red.index[np.arange(0, len(df_red), 10)]))
    dataset_factory.df = dataset_factory.df.loc[np.sort(idx)]
    return resize_dataset(dataset_factory, new_root, img_load="full", **kwargs)

def prepare_atrw(root, new_root="data/ATRW", **kwargs):
    dataset_factory = datasets.ATRW(root)
    return resize_dataset(dataset_factory, new_root, img_load="bbox", **kwargs)

def prepare_beluga_id(root, new_root="data/BelugaID", **kwargs):
    dataset_factory = datasets.BelugaIDv2(root)
    return resize_dataset(dataset_factory, new_root, img_load="bbox", **kwargs)

def prepare_bird_individual_id(root, new_root="data/BirdIndividualID", segmented=True, **kwargs):
    if segmented:
        root = root + "Segmented"
    dataset_factory = datasets.BirdIndividualIDSegmented(root)
    return resize_dataset(dataset_factory, new_root, img_load="crop_black", **kwargs)

def prepare_cat_individual_images(root, new_root="data/CatIndividualImages", **kwargs):
    dataset_factory = datasets.CatIndividualImages(root)
    return resize_dataset(dataset_factory, new_root, img_load="full", **kwargs)

def prepare_cow_dataset(root, new_root="data/CowDataset", **kwargs):
    dataset_factory = datasets.CowDataset(root)
    return resize_dataset(dataset_factory, new_root, img_load="full", **kwargs)

def prepare_cows2021(root, new_root="data/Cows2021", **kwargs):
    dataset_factory = datasets.Cows2021v2(root)
    return resize_dataset(dataset_factory, new_root, img_load="full", **kwargs)

def prepare_ctai(root, new_root="data/CTai", **kwargs):
    dataset_factory = datasets.CTai(root)
    return resize_dataset(dataset_factory, new_root, img_load="full", **kwargs)

def prepare_czoo(root, new_root="data/CZoo", **kwargs):
    dataset_factory = datasets.CZoo(root)
    return resize_dataset(dataset_factory, new_root, img_load="full", **kwargs)

def prepare_dog_facenet(root, new_root="data/DogFaceNet", **kwargs):
    dataset_factory = datasets.DogFaceNet(root)
    return resize_dataset(dataset_factory, new_root, img_load="full", **kwargs)

def prepare_friesian_cattle_2015(root, new_root="data/FriesianCattle2015", **kwargs):
    dataset_factory = datasets.FriesianCattle2015v2(root)
    return resize_dataset(dataset_factory, new_root, img_load="crop_black", **kwargs)

def prepare_friesian_cattle_2017(root, new_root="data/FriesianCattle2017", **kwargs):
    dataset_factory = datasets.FriesianCattle2017(root)
    return resize_dataset(dataset_factory, new_root, img_load="full", **kwargs)

def prepare_giraffes(root, new_root="data/Giraffes", **kwargs):
    dataset_factory = datasets.Giraffes(root)
    return resize_dataset(dataset_factory, new_root, img_load="full", **kwargs)

def prepare_giraffe_zebra_id(root, new_root="data/GiraffeZebraID", **kwargs):
    dataset_factory = datasets.GiraffeZebraID(root)
    return resize_dataset(dataset_factory, new_root, img_load="bbox", **kwargs)

def prepare_hyena_id_2022(root, new_root="data/HyenaID2022", **kwargs):
    dataset_factory = datasets.HyenaID2022(root)
    return resize_dataset(dataset_factory, new_root, img_load="bbox", **kwargs)

def prepare_ipanda_50(root, new_root="data/IPanda50", **kwargs):
    dataset_factory = datasets.IPanda50(root)
    return resize_dataset(dataset_factory, new_root, img_load="full", **kwargs)

def prepare_leopard_id_2022(root, new_root="data/LeopardID2022", **kwargs):
    dataset_factory = datasets.LeopardID2022(root)
    return resize_dataset(dataset_factory, new_root, img_load="bbox", **kwargs)

def prepare_mpdd(root, new_root="data/MPDD", **kwargs):
    dataset_factory = datasets.MPDD(root)
    return resize_dataset(dataset_factory, new_root, img_load="full", **kwargs)

def prepare_ndd20(root, new_root="data/NDD20", **kwargs):
    dataset_factory = datasets.NDD20v2(root)
    return resize_dataset(dataset_factory, new_root, img_load="bbox_mask", **kwargs)

def prepare_nyala_data(root, new_root="data/NyalaData", **kwargs):
    dataset_factory = datasets.NyalaData(root)
    return resize_dataset(dataset_factory, new_root, img_load="full", **kwargs)

def prepare_open_cows_2020(root, new_root="data/OpenCows2020", **kwargs):
    dataset_factory = datasets.OpenCows2020(root)
    return resize_dataset(dataset_factory, new_root, img_load="full", **kwargs)

def prepare_polar_bear_vidid(root, new_root="data/PolarBearVidID", **kwargs):
    dataset_factory = datasets.PolarBearVidID(root)
    # Take only every tenth frame in videos
    idx = np.array([], dtype=int)
    for _, df_red in dataset_factory.df.groupby(['identity', 'video']):
        idx = np.hstack((idx, df_red.index[np.arange(0, len(df_red), 10)]))
    dataset_factory.df = dataset_factory.df.loc[np.sort(idx)]
    return resize_dataset(dataset_factory, new_root, img_load="full", **kwargs)

def prepare_seal_id(root, new_root="data/SealID", segmented=True, **kwargs):
    if segmented:
        root = root + "Segmented"
    dataset_factory = datasets.SealIDSegmented(root)
    return resize_dataset(dataset_factory, new_root, img_load="crop_black", **kwargs)

def prepare_sea_star_reid_2023(root, new_root="data/SeaStarReID2023", **kwargs):
    dataset_factory = datasets.SeaStarReID2023(root)
    return resize_dataset(dataset_factory, new_root, img_load="full", **kwargs)

def prepare_sea_turtle_id_2022(root, new_root="data/SeaTurtleID2022", **kwargs):
    dataset_factory = datasets.SeaTurtleID2022(root)
    return resize_dataset(dataset_factory, new_root, img_load="bbox", **kwargs)

def prepare_smalst(root, new_root="data/SMALST", size=None, copy_files=True):
    dataset_factory = datasets.SMALST(root)
    dataset = WildlifeDataset(
        dataset_factory.df,
        dataset_factory.root,
        img_load="full",
    )
    dataset_masks = WildlifeDataset(
        dataset_factory.df,
        dataset_factory.root,
        img_load="full",
        col_path="segmentation",
    )

    df_new = []
    for i in tqdm(range(len(dataset))):
        row = dataset.metadata.iloc[i]
        img_path = row["path"].replace('\\\\', '/')
        img_path = img_path.replace('\\', '/')

        if copy_files:
            if not os.path.exists(os.path.dirname(os.path.join(new_root, img_path))):
                os.makedirs(os.path.dirname(os.path.join(new_root, img_path)))

            # Apply mask
            img, _ = dataset[i]
            mask, _ = dataset_masks[i]
            img = Image.fromarray(img * np.array(mask).astype(bool))

            # Crop black parts and resize
            y_nonzero, x_nonzero, _ = np.nonzero(img)
            img = img.crop(
                (np.min(x_nonzero), np.min(y_nonzero), np.max(x_nonzero), np.max(y_nonzero))
            )
            if size is not None:
                img = T.Resize(size=size)(img)

            # Save image
            img.save(os.path.join(new_root, img_path))

        # Update dataframe
        df_new.append({
            'image_id': row['image_id'],
            'identity': row['identity'],
            'path': img_path,
            'species': 'zebra',
            'date': row.get('date', np.nan),
            'orientation': row.get('orientation', np.nan),
        })
    return pd.DataFrame(df_new)

def prepare_stripe_spotter(root, new_root="data/StripeSpotter", **kwargs):
    dataset_factory = datasets.StripeSpotter(root)
    return resize_dataset(dataset_factory, new_root, img_load="bbox", **kwargs)

def prepare_whaleshark_id(root, new_root="data/WhaleSharkID", **kwargs):
    dataset_factory = datasets.WhaleSharkID(root)
    return resize_dataset(dataset_factory, new_root, img_load="bbox", **kwargs)

def prepare_zindi_turtle_recall(root, new_root="data/ZindiTurtleRecall", **kwargs):
    dataset_factory = datasets.ZindiTurtleRecall(root)
    return resize_dataset(dataset_factory, new_root, img_load="full", **kwargs)

prepare_functions = {
    'AAUZebraFish': prepare_aau_zebrafish,
    'AerialCattle2017': prepare_aerial_cattle_2017,
    'ATRW': prepare_atrw,
    'BelugaID': prepare_beluga_id,
    'BirdIndividualID': prepare_bird_individual_id,
    'CatIndividualImages': prepare_cat_individual_images,
    'CowDataset': prepare_cow_dataset,
    'Cows2021': prepare_cows2021,
    'CTai': prepare_ctai,
    'CZoo': prepare_czoo,
    'DogFaceNet': prepare_dog_facenet,
    'FriesianCattle2015': prepare_friesian_cattle_2015,
    'FriesianCattle2017': prepare_friesian_cattle_2017,
    'Giraffes': prepare_giraffes,
    'GiraffeZebraID': prepare_giraffe_zebra_id,
    'HyenaID2022': prepare_hyena_id_2022,
    'IPanda50': prepare_ipanda_50,
    'LeopardID2022': prepare_leopard_id_2022,
    'MPDD': prepare_mpdd,
    'NDD20': prepare_ndd20,
    'NyalaData': prepare_nyala_data,
    'OpenCows2020': prepare_open_cows_2020,
    'PolarBearVidID': prepare_polar_bear_vidid,
    'SealID': prepare_seal_id,
    'SeaStarReID2023': prepare_sea_star_reid_2023,
    'SeaTurtleID2022': prepare_sea_turtle_id_2022,
    'SMALST': prepare_smalst,
    'StripeSpotter': prepare_stripe_spotter,
    'WhaleSharkID': prepare_whaleshark_id,
    'ZindiTurtleRecall': prepare_zindi_turtle_recall,
}

species_conversion = {
    'Anthenea australiae': 'sea star',
    'Asteria rubens': 'sea star',
    'BND': 'doplhin',
    'Friesian cattle': 'cow',
    'WBD': 'doplhin',
    'amur tiger': 'tiger',
    'beluga whale': 'whale',
    'cat': 'cat',
    'chimpanzee': 'chimpanzee',
    'cow': 'cow',
    'dog': 'dog',
    'giraffe': 'giraffe',
    'giraffe_masai': 'giraffe',
    'great panda': 'panda',
    'great_tits': 'bird',
    'leopard': 'leopard',
    'loggerhead turtle': 'sea turtle',
    'nyala': 'nyala',
    'polar bear': 'polar bear',
    'ringed seal': 'seal',
    'sea turtle': 'sea turtle',
    'sociable_weavers': 'bird',
    'spotted hyena': 'hyena',
    'whale shark': 'shark',
    'zebra': 'zebra',
    'zebra_finch': 'zebra',
    'zebra_plains': 'zebra',
    'zebrafish': 'fish',
}
