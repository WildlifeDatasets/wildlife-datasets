import os
scripts_folder = 'download_scripts'
scripts = os.listdir(scripts_folder)

scripts = [
    # Easy datasets
    'aau_zebrafish_id.py',
    'aerial_cattle_2017.py',
    'bird_individual_id.py',
    'c_tai.py',
    'c_zoo.py',
    'cows_2021.py',
    'friesian_cattle_2015.py',
    'friesian_cattle_2017.py',
    'giraffes.py',
    'ipanda_50.py',
    'lion_data.py',
    'macaque_faces.py',
    'ndd20.py',
    'noaa_right_whale.py',
    'nyala_data.py',
    'open_cows_2020.py',
    'seal_id.py',
    'smalst.py',
    'stripe_spotter.py',
    'zindi_turtle_recall.py',

    # LILA BC datasets
    'beluga_id.py',
    'giraffe_zebra_id.py',
    'hyena_id_2022.py',
    'leopard_id_2022.py',
    'whale_shark_id.py',

    # Hard - large datasets
    'atrw.py',
    'humpback_whale.py'
    'drosophila.py',
    'happy_whale.py',

    # Hard - very large datasets
    'wni_giraffes.py',
]

os.chdir(scripts_folder)
for script in scripts:
    print('\n')
    print(f'Downloading: {script}')
    os.system(f"python3 {script}")