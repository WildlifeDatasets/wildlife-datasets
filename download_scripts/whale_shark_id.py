import os
url = 'https://lilablobssc.blob.core.windows.net/whale-shark-id/whaleshark.coco.tar.gz'
name = 'WhaleSharkID'
os.system(f"wget -P '../datasets/{name}' {url}")