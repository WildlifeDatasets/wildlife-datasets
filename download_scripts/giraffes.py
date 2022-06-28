import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=str, default='../datasets',  help="Output folder")
parser.add_argument("--name", type=str, default='Giraffes',  help="Dataset name")
args = parser.parse_args()

directory = os.path.join(args.output, args.name)
if not os.path.exists(directory):
    os.makedirs(directory)
os.chdir(directory)

# Download
url = 'ftp://pbil.univ-lyon1.fr/pub/datasets/miele2021/'
os.system(f"wget -rpk -l 10 -np -c --random-wait -U Mozilla {url} -P '.' ")