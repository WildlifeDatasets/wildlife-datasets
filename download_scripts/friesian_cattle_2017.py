import os
url = 'https://data.bris.ac.uk/datasets/2yizcfbkuv4352pzc32n54371r/2yizcfbkuv4352pzc32n54371r.zip'
name = 'FriesianCattle2017'

os.system(f"wget -P '../datasets/{name}' {url}")