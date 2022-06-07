import os
import argparse
parser = argparse.ArgumentParser(description='')
parser.add_argument("--output", type=str, default='../datasets',  help="Output folder")
parser.add_argument("--name", type=str, default='StripeSpotter',  help="Dataset name")
args = parser.parse_args()

directory = os.path.join(args.output, args.name)
if not os.path.exists(directory):
    os.makedirs(directory)
os.chdir(directory)

# Download
urls = [
    'https://storage.googleapis.com/google-code-archive-downloads/v2/code.google.com/stripespotter/data-20110718.zip',
    'https://storage.googleapis.com/google-code-archive-downloads/v2/code.google.com/stripespotter/data-20110718.z02',
    'https://storage.googleapis.com/google-code-archive-downloads/v2/code.google.com/stripespotter/data-20110718.z01',
    ]
for url in urls:
    os.system(f"wget -P '.' {url}")

# Unpack
os.system(f"zip -s- data-20110718.zip -O data-full.zip")
os.system(f"unzip data-full.zip")

# Cleanup
os.remove('data-20110718.zip')
os.remove('data-20110718.z01')
os.remove('data-20110718.z02')
os.remove('data-full.zip')
