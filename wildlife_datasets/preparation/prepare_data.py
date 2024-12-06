import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from .. import datasets
from typing import Optional, List

def resize_dataset(
        dataset: datasets.DatasetFactory,
        new_root: str,
        idx: Optional[List[int]] = None,
        copy_files: bool = True
        ) -> pd.DataFrame:
    """Resizes dataset using `dataset.transform` into `new_root`.

    Args:
        dataset (datasets.DatasetFactory): Dataset to be resized.
        new_root (str): Root to store new images.
        idx (Optional[List[int]], optional): If specified, then indices to consider.
        copy_files (bool, optional): Whether files should be copied as well or only datatframe created.

    Returns:
        Description of the new dataset.
    """
    
    if idx is None:
        idx = range(len(dataset))
    
    df_new = []
    for i in tqdm(idx, mininterval=1, ncols=100):
        # Make image path unique across datasets
        row = dataset.metadata.iloc[i]
        base, ext = os.path.splitext(row[dataset.col_path])
        img_path = base + "_" + str(row["image_id"]) + ext
        img_path = img_path.replace('\\\\', '/')
        img_path = img_path.replace('\\', '/')

        # Save image to new root with unique image path
        if copy_files:
            image = dataset[i]
            full_img_path = os.path.join(new_root, img_path)
            if not os.path.exists(os.path.dirname(full_img_path)):
                os.makedirs(os.path.dirname(full_img_path))
            image.save(full_img_path)

        # Extract species
        if 'species' in row:
            species = row['species']
        else:
            species = dataset.summary['animals']
            if len(species) != 1:
                raise Exception('There should be only one species')
            species = list(species)[0]

        # Update dataframe
        df_new.append({
            'image_id': row['image_id'],
            dataset.col_label: row[dataset.col_label],
            dataset.col_path: img_path,
            'species': species,
            'date': row.get('date', np.nan),
            'orientation': row.get('orientation', np.nan),
        })
    return pd.DataFrame(df_new)

def get_every_k(
        dataset: datasets.DatasetFactory,
        k: int,
        groupby_cols: str | List[str],
        ) -> List[int]:
    """Gets indices of every k-th image based on columns in `groupby_cols`.

    Args:
        dataset (datasets.DatasetFactory): Dataset to be resized.
        k (int): Number of images to skip.
        groupby_cols (str | List[str]): For which groups the indices will be computed.

    Returns:
        Computed indices.
    """
    
    idx = np.array([], dtype=int)
    for _, df_red in dataset.df.groupby(groupby_cols):
        idx = np.hstack((idx, df_red.index[np.arange(0, len(df_red), k)]))
    # Convert loc to iloc
    idx = dataset.df.index.get_indexer(idx)
    return np.sort(idx)

def prepare_aau_zebrafish(root, new_root, transform=None, **kwargs):
    dataset = datasets.AAUZebraFish(root, img_load="bbox", transform=transform, remove_unknown=True)
    idx = get_every_k(dataset, 20, dataset.col_label)
    return resize_dataset(dataset, new_root, idx=idx, **kwargs)

def prepare_aerial_cattle_2017(root, new_root, transform=None, **kwargs):
    dataset = datasets.AerialCattle2017(root, img_load="full", transform=transform, remove_unknown=True)
    idx = get_every_k(dataset, 20, dataset.col_label)
    return resize_dataset(dataset, new_root, idx=idx, **kwargs)

def prepare_amvrakikos_turtles(root, new_root, transform=None, **kwargs):
    dataset = datasets.AmvrakikosTurtles(root, img_load="bbox", transform=transform, remove_unknown=True)
    return resize_dataset(dataset, new_root, **kwargs)

def prepare_atrw(root, new_root, transform=None, **kwargs):
    dataset = datasets.ATRW(root, img_load="bbox", transform=transform, remove_unknown=True)
    return resize_dataset(dataset, new_root, **kwargs)

def prepare_beluga_id(root, new_root, transform=None, **kwargs):
    dataset = datasets.BelugaIDv2(root, img_load="bbox", transform=transform, remove_unknown=True)
    return resize_dataset(dataset, new_root, **kwargs)

def prepare_bird_individual_id(root, new_root, transform=None, segmented=True, **kwargs):
    if segmented:
        root = root + "Segmented"
    dataset = datasets.BirdIndividualIDSegmented(root, img_load="crop_black", transform=transform, remove_unknown=True)
    idx = get_every_k(dataset, 20, dataset.col_label)
    return resize_dataset(dataset, new_root, idx=idx, **kwargs)

def prepare_cat_individual_images(root, new_root, transform=None, **kwargs):
    dataset = datasets.CatIndividualImages(root, img_load="full", transform=transform, remove_unknown=True)
    return resize_dataset(dataset, new_root, **kwargs)

def prepare_chicks4free_id(root, new_root, transform=None, **kwargs):
    dataset = datasets.Chicks4FreeID(root, img_load="full", transform=transform, remove_unknown=True)
    # Change the path from np.nan so that it is saved correctly
    dataset.df[dataset.col_path] = 'images/' + dataset.df['image_id'].astype('str') + '.jpg'
    return resize_dataset(dataset, new_root, **kwargs)

def prepare_cow_dataset(root, new_root, transform=None, **kwargs):
    dataset = datasets.CowDataset(root, img_load="full", transform=transform, remove_unknown=True)
    return resize_dataset(dataset, new_root, **kwargs)

def prepare_cows2021(root, new_root, transform=None, **kwargs):
    dataset = datasets.Cows2021v2(root, img_load="full", transform=transform, remove_unknown=True)
    return resize_dataset(dataset, new_root, **kwargs)

def prepare_ctai(root, new_root, transform=None, **kwargs):
    dataset = datasets.CTai(root, img_load="full", transform=transform, remove_unknown=True)
    return resize_dataset(dataset, new_root, **kwargs)

def prepare_czoo(root, new_root, transform=None, **kwargs):
    dataset = datasets.CZoo(root, img_load="full", transform=transform, remove_unknown=True)
    return resize_dataset(dataset, new_root, **kwargs)

def prepare_dog_facenet(root, new_root, transform=None, **kwargs):
    dataset = datasets.DogFaceNet(root, img_load="full", transform=transform, remove_unknown=True)
    return resize_dataset(dataset, new_root, **kwargs)

def prepare_drosophila(root, new_root, transform=None, **kwargs):
    dataset = datasets.Drosophila(root, img_load="full", transform=transform, remove_unknown=True)
    idx = get_every_k(dataset, 1000, dataset.col_label)
    return resize_dataset(dataset, new_root, idx=idx, **kwargs)

def prepare_friesian_cattle_2015(root, new_root, transform=None, **kwargs):
    dataset = datasets.FriesianCattle2015v2(root, img_load="crop_black", transform=transform, remove_unknown=True)
    return resize_dataset(dataset, new_root, **kwargs)

def prepare_friesian_cattle_2017(root, new_root, transform=None, **kwargs):
    dataset = datasets.FriesianCattle2017(root, img_load="full", transform=transform, remove_unknown=True)
    return resize_dataset(dataset, new_root, **kwargs)

def prepare_giraffes(root, new_root, transform=None, **kwargs):
    dataset = datasets.Giraffes(root, img_load="full", transform=transform, remove_unknown=True)
    return resize_dataset(dataset, new_root, **kwargs)

def prepare_giraffe_zebra_id(root, new_root, transform=None, **kwargs):
    dataset = datasets.GiraffeZebraID(root, img_load="bbox", transform=transform, remove_unknown=True)
    return resize_dataset(dataset, new_root, **kwargs)

def prepare_happy_whale(root, new_root, transform=None, **kwargs):
    dataset = datasets.HappyWhale(root, img_load="full", transform=transform, remove_unknown=True)
    return resize_dataset(dataset, new_root, **kwargs)

def prepare_humpback_whale_id(root, new_root, transform=None, **kwargs):
    dataset = datasets.HumpbackWhaleID(root, img_load="full", transform=transform, remove_unknown=True)
    return resize_dataset(dataset, new_root, **kwargs)

def prepare_hyena_id_2022(root, new_root, transform=None, **kwargs):
    dataset = datasets.HyenaID2022(root, img_load="bbox", transform=transform, remove_unknown=True)
    return resize_dataset(dataset, new_root, **kwargs)

def prepare_ipanda_50(root, new_root, transform=None, **kwargs):
    dataset = datasets.IPanda50(root, img_load="full", transform=transform, remove_unknown=True)
    return resize_dataset(dataset, new_root, **kwargs)

def prepare_leopard_id_2022(root, new_root, transform=None, **kwargs):
    dataset = datasets.LeopardID2022(root, img_load="bbox", transform=transform, remove_unknown=True)
    return resize_dataset(dataset, new_root, **kwargs)

def prepare_macaque_faces(root, new_root, transform=None, **kwargs):
    dataset = datasets.MacaqueFaces(root, img_load="full", transform=transform, remove_unknown=True)
    return resize_dataset(dataset, new_root, **kwargs)

def prepare_mpdd(root, new_root, transform=None, **kwargs):
    dataset = datasets.MPDD(root, img_load="full", transform=transform, remove_unknown=True)
    return resize_dataset(dataset, new_root, **kwargs)

def prepare_ndd20(root, new_root, transform=None, **kwargs):
    dataset = datasets.NDD20v2(root, img_load="bbox_mask", transform=transform, remove_unknown=True)
    return resize_dataset(dataset, new_root, **kwargs)

def prepare_noaa_right_whale(root, new_root, transform=None, **kwargs):
    dataset = datasets.NOAARightWhale(root, img_load="full", transform=transform, remove_unknown=True)
    return resize_dataset(dataset, new_root, **kwargs)

def prepare_nyala_data(root, new_root, transform=None, **kwargs):
    dataset = datasets.NyalaData(root, img_load="full", transform=transform, remove_unknown=True)
    return resize_dataset(dataset, new_root, **kwargs)

def prepare_open_cows_2020(root, new_root, transform=None, **kwargs):
    dataset = datasets.OpenCows2020(root, img_load="full", transform=transform, remove_unknown=True)
    return resize_dataset(dataset, new_root, **kwargs)

def prepare_polar_bear_vidid(root, new_root, transform=None, **kwargs):
    dataset = datasets.PolarBearVidID(root, img_load="full", transform=transform, remove_unknown=True)
    idx = get_every_k(dataset, 100, dataset.col_label)
    return resize_dataset(dataset, new_root, idx=idx, **kwargs)

def prepare_reunion_turtles(root, new_root, transform=None, **kwargs):
    dataset = datasets.ReunionTurtles(root, img_load="full", transform=transform, remove_unknown=True)
    return resize_dataset(dataset, new_root, **kwargs)

def prepare_seal_id(root, new_root, transform=None, segmented=True, **kwargs):
    if segmented:
        root = root + "Segmented"
    dataset = datasets.SealIDSegmented(root, img_load="crop_black", transform=transform, remove_unknown=True)
    return resize_dataset(dataset, new_root, **kwargs)

def prepare_sea_star_reid_2023(root, new_root, transform=None, **kwargs):
    dataset = datasets.SeaStarReID2023(root, img_load="full", transform=transform, remove_unknown=True)
    return resize_dataset(dataset, new_root, **kwargs)

def prepare_sea_turtle_id_2022(root, new_root, transform=None, **kwargs):
    dataset = datasets.SeaTurtleID2022(root, img_load="bbox", transform=transform, remove_unknown=True)
    return resize_dataset(dataset, new_root, **kwargs)

def prepare_smalst(root, new_root, transform=None, **kwargs):
    dataset = datasets.SMALST(root, img_load="bbox_mask", transform=transform, remove_unknown=True)
    idx = get_every_k(dataset, 10, dataset.col_label)
    return resize_dataset(dataset, new_root, idx=idx, **kwargs)

def prepare_southern_province_turtles(root, new_root, transform=None, **kwargs):
    dataset = datasets.SouthernProvinceTurtles(root, img_load="full", transform=transform, remove_unknown=True)
    return resize_dataset(dataset, new_root, **kwargs)

def prepare_stripe_spotter(root, new_root, transform=None, **kwargs):
    dataset = datasets.StripeSpotter(root, img_load="bbox", transform=transform, remove_unknown=True)
    return resize_dataset(dataset, new_root, **kwargs)

def prepare_whaleshark_id(root, new_root, transform=None, **kwargs):
    dataset = datasets.WhaleSharkID(root, img_load="bbox", transform=transform, remove_unknown=True)
    return resize_dataset(dataset, new_root, **kwargs)

def prepare_zakynthos_turtles(root, new_root, transform=None, **kwargs):
    dataset = datasets.ZakynthosTurtles(root, img_load="bbox", transform=transform, remove_unknown=True)
    return resize_dataset(dataset, new_root, **kwargs)

def prepare_zindi_turtle_recall(root, new_root, transform=None, **kwargs):
    dataset = datasets.ZindiTurtleRecall(root, img_load="full", transform=transform, remove_unknown=True)
    return resize_dataset(dataset, new_root, **kwargs)

prepare_functions = {
    'AAUZebraFish': prepare_aau_zebrafish,
    'AerialCattle2017': prepare_aerial_cattle_2017,
    'AmvrakikosTurtles': prepare_amvrakikos_turtles,
    'ATRW': prepare_atrw,
    'BelugaID': prepare_beluga_id,
    'BirdIndividualID': prepare_bird_individual_id,
    'CatIndividualImages': prepare_cat_individual_images,
    'Chicks4FreeID': prepare_chicks4free_id,
    'CowDataset': prepare_cow_dataset,
    'Cows2021': prepare_cows2021,
    'CTai': prepare_ctai,
    'CZoo': prepare_czoo,
    'DogFaceNet': prepare_dog_facenet,
    'Drosophila': prepare_drosophila,
    'FriesianCattle2015': prepare_friesian_cattle_2015,
    'FriesianCattle2017': prepare_friesian_cattle_2017,
    'Giraffes': prepare_giraffes,
    'GiraffeZebraID': prepare_giraffe_zebra_id,
    'HappyWhale': prepare_happy_whale,
    'HumpbackWhaleID': prepare_humpback_whale_id,
    'HyenaID2022': prepare_hyena_id_2022,
    'IPanda50': prepare_ipanda_50,
    'LeopardID2022': prepare_leopard_id_2022,
    'MacaqueFaces': prepare_macaque_faces,
    'MPDD': prepare_mpdd,
    'NDD20': prepare_ndd20,
    'NOAARightWhale': prepare_noaa_right_whale,
    'NyalaData': prepare_nyala_data,
    'OpenCows2020': prepare_open_cows_2020,
    'PolarBearVidID': prepare_polar_bear_vidid,
    'ReunionTurtles': prepare_reunion_turtles,
    'SealID': prepare_seal_id,
    'SeaStarReID2023': prepare_sea_star_reid_2023,
    'SeaTurtleID2022': prepare_sea_turtle_id_2022,
    'SMALST': prepare_smalst,
    'SouthernProvinceTurtles': prepare_southern_province_turtles,
    'StripeSpotter': prepare_stripe_spotter,
    'WhaleSharkID': prepare_whaleshark_id,
    'ZakynthosTurtles': prepare_zakynthos_turtles,
    'ZindiTurtleRecall': prepare_zindi_turtle_recall,
}

species_conversion = {
    'Anthenea australiae': 'sea star',
    'Asteria rubens': 'sea star',
    'bottlenose_dolphin': 'dolphin',
    'beluga': 'whale',
    'blue_whale': 'whale',
    'BND': 'dolphin',
    'Friesian cattle': 'cow',
    'WBD': 'dolphin',
    'amur tiger': 'tiger',
    'beluga whale': 'whale',
    'brydes_whale': 'whale',
    'cat': 'cat',
    'chickens': 'chicken',
    'chimpanzee': 'chimpanzee',
    'commersons_dolphin': 'dolphin',
    'common_dolphin': 'dolphin',
    'cow': 'cow',
    'cuviers_beaked_whale': 'whale',
    'dog': 'dog',
    'drosophila': 'fly',
    'dusky_dolphin': 'dolphin',
    'false_killer_whale': 'whale',
    'fin_whale': 'whale',
    'frasiers_dolphin': 'dolphin',
    'giraffe': 'giraffe',
    'giraffe_masai': 'giraffe',
    'globis': 'whale',
    'gray_whale': 'whale',
    'great panda': 'panda',
    'great_tits': 'bird',
    'green turtle': 'sea turtle',
    'Green': 'sea turtle',
    'Hawksbill': 'sea turtle',
    'humpback_whale': 'whale',
    'killer_whale': 'whale',
    'leopard': 'leopard',
    'loggerhead turtle': 'sea turtle',
    'long_finned_pilot_whale': 'whale',
    'melon_headed_whale': 'whale',
    'minke_whale': 'whale',
    'nyala': 'nyala',
    'pantropic_spotted_dolphin': 'dolphin',
    'pilot_whale': 'whale',
    'polar bear': 'polar bear',
    'pygmy_killer_whale': 'whale',
    'right whale': 'whale',
    'ringed seal': 'seal',
    'rhesus macaque': 'macaque',
    'rough_toothed_dolphin': 'dolphin',
    'sea turtle': 'sea turtle',
    'sei_whale': 'whale',
    'short_finned_pilot_whale': 'whale',
    'sociable_weavers': 'bird',
    'southern_right_whale': 'whale',
    'spinner_dolphin': 'dolphin',
    'spotted_dolphin': 'whale',
    'spotted hyena': 'hyena',
    'whale': 'whale',
    'whale shark': 'shark',
    'white_sided_dolphin': 'dolphin',
    'zebra': 'zebra',
    'zebra_finch': 'bird',
    'zebra_plains': 'zebra',
    'zebrafish': 'fish',
}
