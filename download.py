import os
scripts_folder = 'download_scripts'
scripts = os.listdir(scripts_folder)

scripts = [
    # Easy datasets
    'lion_data.py',
    'nyala_data.py',
    'c_zoo.py',
    'c_tai.py',
    'aau_zebrafish_id.py',
    'macaque_faces.py',
    'stripe_spotter.py',
    'zindi_turtle_recall.py',
    'ipanda_50.py',

    # LILA BC datasets
    'whale_shark_id.py',
    'giraffe_zebra_id.py',
    'leopard_id_2022.py',
    'hyena_id_2022.py',
    'beluga_id.py',

    # Giraffes
    'wni_giraffes.py',

    # Cattle
    'cows_2021.py',
    'aerial_cattle_2017.py',
    'open_cows_2020.py',
    'friesian_cattle_2017.py',
    'friesian_cattle_2015.py',

    # Hard - large datasets
    'drosophila.py',
    'atrw.py',
    'happy_whale.py',

]

os.chdir(scripts_folder)
for script in scripts:
    print('\n')
    print(f'Downloading: {script}')
    os.system(f"python3 {script}")