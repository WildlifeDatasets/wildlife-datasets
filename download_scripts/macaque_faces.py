import os
name = 'MacaqueFaces'
urls = [
    'https://github.com/clwitham/MacaqueFaces/raw/master/ModelSet/MacaqueFaces.zip',
    'https://github.com/clwitham/MacaqueFaces/raw/master/ModelSet/MacaqueFaces_ImageInfo.csv',
]
for url in urls:
    os.system(f"wget -P '../datasets/{name}' {url}")