import os
name = 'GiraffeZebraID'
url = 'https://lilablobssc.blob.core.windows.net/giraffe-zebra-id/gzgc.coco.tar.gz'
os.system(f"wget -P '../datasets/{name}' {url}")