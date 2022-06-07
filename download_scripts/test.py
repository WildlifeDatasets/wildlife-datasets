#import os
#name = 'Test'
#url = "http://www.gutenberg.lib.md.us/4/8/8/2/48824/48824-8.zip"
#os.system(f"wget -P '../datasets/{name}' {url}")


import os
import argparse
parser = argparse.ArgumentParser(description='')
parser.add_argument("--output", type=str, default='../datasets',  help="Output folder")
args = parser.parse_args()

name = 'StripeSpotter'
directory = os.path.join(args.output, name)
if not os.path.exists(directory):
    os.makedirs(directory)
os.chdir(directory)

# Download
urls = [
    'https://storage.googleapis.com/google-code-archive-downloads/v2/code.google.com/stripespotter/data-20110718.zip',
    'https://storage.googleapis.com/google-code-archive-downloads/v2/code.google.com/stripespotter/data-20110718.z02',
    'https://storage.googleapis.com/google-code-archive-downloads/v2/code.google.com/stripespotter/data-20110718.z01',
    ]
#for url in urls:
#    os.system(f"wget -P '.' {url}")


#os.remove('48824-8.zip')

import shutil
shutil.unpack_archive('data-20110718.zip')

print('Extracting')
data = ['data-20110718.zip', 'data-20110718.z01', 'data-20110718.z02']

with open('output_file.zip','wb') as wfd:
    for f in data: # Search for all files matching searchstring
        with open(f,'rb') as fd:
            shutil.copyfileobj(fd, wfd) # Concatinate

import zipfile
with zipfile.ZipFile('output_file.zip', 'r') as zip_ref:
    zip_ref.extractall('.')