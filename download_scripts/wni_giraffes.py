import os
import argparse
import utils
import shutil

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=str, default='../datasets',  help="Output folder")
parser.add_argument("--name", type=str, default='WNIGiraffes',  help="Dataset name")
args = parser.parse_args()

directory = os.path.join(args.output, args.name)
if not os.path.exists(directory):
    os.makedirs(directory)

shutil.copy('../azcopy', os.path.join(directory, 'azcopy'))
os.chdir(directory)


# Images
url = "https://lilablobssc.blob.core.windows.net/wni-giraffes/wni_giraffes_train_images.zip"
archive = 'wni_giraffes_train_images.zip'
os.system(f'./azcopy cp {url} {archive}')
utils.extract_archive(archive, delete=True)

# Metadata
url = 'https://lilablobssc.blob.core.windows.net/wni-giraffes/wni_giraffes_train.zip'
archive = 'wni_giraffes_train.zip'
utils.download_url(url, archive)
utils.extract_archive(archive, delete=True)