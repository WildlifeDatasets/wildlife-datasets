import os
url = 'https://data.bris.ac.uk/datasets/tar/3owflku95bxsx24643cybxu3qh.zip'
name = 'AerialCattle2017'

os.system(f"wget -P '../datasets/{name}' {url}")