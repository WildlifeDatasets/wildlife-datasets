import os
url = 'https://lilablobssc.blob.core.windows.net/liladata/wild-me/hyena.coco.tar.gz'
name = 'HyenaID2022'
os.system(f"wget -P '../datasets/{name}' {url}")