import os
url = 'https://data.bris.ac.uk/datasets/tar/10m32xl88x2b61zlkkgz3fml17.zip'
name = 'OpenCow2020'

os.system(f"wget -P '../datasets/{name}' {url}")