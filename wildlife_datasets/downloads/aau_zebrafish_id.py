import os
import argparse
from . import utils

def get_data(root):
    with utils.data_directory(root):
        os.system(f"kaggle datasets download -d 'aalborguniversity/aau-zebrafish-reid' --unzip")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default='data',  help="Output folder")
    parser.add_argument("--name", type=str, default='AAUZebraFishID',  help="Dataset name")
    args = parser.parse_args()
    get_data(os.path.join(args.output, args.name))