import os
url = 'https://github.com/cvjena/chimpanzee_faces/archive/refs/heads/master.zip'
name = 'ChimpanzeeFaces'
# SHOULD BE TWO DATASETS

os.system(f"wget -P '../datasets/{name}' {url}")