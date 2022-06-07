import os
import argparse
import gdown
import zipfile

parser = argparse.ArgumentParser(description='')
parser.add_argument("--output", type=str, default='../datasets',  help="Output folder")
args = parser.parse_args()

name = 'IPanda-50'
directory = os.path.join(args.output, name)
if not os.path.exists(directory):
    os.makedirs(directory)
os.chdir(directory)

downloads = [
    ('iPanda50-images.zip', 'https://drive.google.com/u/0/uc?id=1nkh-g6a8JvWy-XsMaZqrN2AXoPlaXuFg'),
    ('iPanda50-split.zip', 'https://drive.google.com/uc?id=1gVREtFWkNec4xwqOyKkpuIQIyWU_Y_Ob'),
    ('iPanda50-eyes-labels.zip', 'https://drive.google.com/uc?id=1jdACN98uOxedZDT-6X3rpbooLAAUEbNY'),
]

for archive, url in downloads:
    gdown.download(url, archive, quiet=False)
    with zipfile.ZipFile(archive, 'r') as zip_ref:
        zip_ref.extractall('.')
    os.remove(archive)
