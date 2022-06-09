import os
import argparse
import utils

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=str, default='../datasets',  help="Output folder")
parser.add_argument("--name", type=str, default='WNIGiraffes',  help="Dataset name")
args = parser.parse_args()

directory = os.path.join(args.output, args.name)
if not os.path.exists(directory):
    os.makedirs(directory)
os.chdir(directory)


# Download and extract
downloads = [
    ('https://lilablobssc.blob.core.windows.net/wni-giraffes/wni_giraffes_train_images.zip', 'wni_giraffes_train_images.zip'),
    ('https://lilablobssc.blob.core.windows.net/wni-giraffes/wni_giraffes_train.zip', 'wni_giraffes_train.zip'),
    ]

for url, archive in downloads:
    utils.download_url(url, archive)
    utils.extract_archive(archive, delete=True)
