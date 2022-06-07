import os
url = 'https://data.bris.ac.uk/datasets/tar/4vnrca7qw1642qlwxjadp87h7.zip'
name = 'Cows2021'
os.system(f"wget -P '../datasets/{name}' {url}")