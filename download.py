import os
scripts_folder = 'download_scripts'
scripts = os.listdir(scripts_folder)

os.chdir(scripts_folder)
for script in scripts:
    print('\n')
    print(f'Downloading: {script}')
    os.system(f"python3 {script}")