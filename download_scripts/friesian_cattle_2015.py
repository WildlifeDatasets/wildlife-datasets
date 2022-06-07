import os
url = 'https://data.bris.ac.uk/datasets/wurzq71kfm561ljahbwjhx9n3/wurzq71kfm561ljahbwjhx9n3.zip'
name = 'FriesianCattle2015'

os.system(f"wget -P '../datasets/{name}' {url}")