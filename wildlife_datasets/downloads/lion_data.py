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
        url = 'https://github.com/tvanzyl/wildlife_reidentification/archive/refs/heads/main.zip'
        archive = 'main.zip'
        utils.download_url(url, archive)
        utils.extract_archive(archive, delete=True)

        # Cleanup
        shutil.rmtree('wildlife_reidentification-main/Nyala_Data_Zero')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default='data',  help="Output folder")
    parser.add_argument("--name", type=str, default='LionData',  help="Dataset name")
    args = parser.parse_args()
    get_data(os.path.join(args.output, args.name))