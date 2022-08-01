import os
import argparse
from . import utils

def get_data(root):
    with utils.data_directory(root):
        directory = os.path.join(args.output, args.name)
        if not os.path.exists(directory):
            os.makedirs(directory)
        os.chdir(directory)

        # Download
        downloads = [
            ('https://storage.googleapis.com/dm-turtle-recall/train.csv', 'train.csv'),
            ('https://storage.googleapis.com/dm-turtle-recall/extra_images.csv', 'extra_images.csv'),
            ('https://storage.googleapis.com/dm-turtle-recall/test.csv', 'test.csv'),
            ('https://storage.googleapis.com/dm-turtle-recall/images.tar', 'images.tar'),
        ]
        for url, file in downloads:
            utils.download_url(url, file)

        # Extract
        utils.extract_archive('images.tar', 'images', delete=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default='data',  help="Output folder")
    parser.add_argument("--name", type=str, default='ZindiTurtleRecall',  help="Dataset name")
    args = parser.parse_args()
    get_data(os.path.join(args.output, args.name))