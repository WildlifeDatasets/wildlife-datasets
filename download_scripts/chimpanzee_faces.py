import os
url = 'https://github.com/cvjena/chimpanzee_faces/archive/refs/heads/master.zip'
name = 'ChimpanzeeFaces'

os.system(f"wget -P '../datasets/{name}' {url}")