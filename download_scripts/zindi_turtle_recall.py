import os
name = 'ZindiTurtleRecall'
urls = [
    'https://storage.googleapis.com/dm-turtle-recall/train.csv',
    'https://storage.googleapis.com/dm-turtle-recall/extra_images.csv',
    'https://storage.googleapis.com/dm-turtle-recall/test.csv',
    'https://storage.googleapis.com/dm-turtle-recall/images.tar',
]
for url in urls:
    os.system(f"wget -P '../datasets/{name}' {url}")