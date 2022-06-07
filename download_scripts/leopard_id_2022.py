import os
url = 'https://lilablobssc.blob.core.windows.net/liladata/wild-me/leopard.coco.tar.gz'
name = 'LeopardID2022'
os.system(f"wget -P '../datasets/{name}' {url}")