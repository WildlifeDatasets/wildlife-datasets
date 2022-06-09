import os
import argparse
import utils

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=str, default='../datasets',  help="Output folder")
parser.add_argument("--name", type=str, default='GiraffeZebraID',  help="Dataset name")
args = parser.parse_args()

directory = os.path.join(args.output, args.name)
if not os.path.exists(directory):
    os.makedirs(directory)
os.chdir(directory)

# Download and extract
url = 'https://lilablobssc.blob.core.windows.net/giraffe-zebra-id/gzgc.coco.tar.gz'
archive = 'gzgc.coco.tar.gz'
utils.download_url(url, archive)
utils.extract_archive(archive, delete=True)
