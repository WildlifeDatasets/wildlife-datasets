import os
import argparse
import shutil
from . import utils

def get_data(root):
    with utils.data_directory(root):
        directory = os.path.join(args.output, args.name)
        if not os.path.exists(directory):
            os.makedirs(directory)
        os.chdir(directory)

        # Download and extract
        url = 'https://github.com/cvjena/chimpanzee_faces/archive/refs/heads/master.zip'
        archive = 'master.zip'
        utils.download_url(url, archive)
        utils.extract_archive(archive, delete=True)

        # Cleanup
        shutil.rmtree('chimpanzee_faces-master/datasets_cropped_chimpanzee_faces/data_CZoo')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default='data',  help="Output folder")
    parser.add_argument("--name", type=str, default='CTai',  help="Dataset name")
    args = parser.parse_args()
    get_data(os.path.join(args.output, args.name))