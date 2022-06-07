import os
name = 'BelugaID'
url = 'https://lilablobssc.blob.core.windows.net/liladata/wild-me/beluga.coco.tar.gz'
os.system(f"wget -P '../datasets/{name}' {url}")