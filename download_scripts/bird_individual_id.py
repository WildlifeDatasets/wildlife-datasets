import os
import argparse
import gdown
import utils

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=str, default='../datasets',  help="Output folder")
parser.add_argument("--name", type=str, default='BirdIndividualID',  help="Dataset name")
args = parser.parse_args()

directory = os.path.join(args.output, args.name)
if not os.path.exists(directory):
    os.makedirs(directory)
os.chdir(directory)

# Try automatic download
#url = 'https://drive.google.com/uc?id=1YT4w8yF44D-y9kdzgF38z2uYbHfpiDOA'
#archive = 'ferreira_et_al_2020.zip'
#gdown.download(url, archive, quiet=False)
#utils.extract_archive(archive, delete=True)

# Download manually from:
url = 'https://drive.google.com/drive/folders/1YkH_2DNVBOKMNGxDinJb97y2T8_wRTZz'

# Upload to kaggle and download
os.system(f"kaggle datasets download -d 'vojtacermak/birds' --unzip")


