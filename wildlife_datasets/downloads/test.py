import os
import argparse
from . import utils

def get_data(root):
    with utils.data_directory(root):
        url = 'https://www.gutenberg.org/ebooks/10.txt.utf-8'
        archive = '10.txt.utf-8'
        utils.download_url(url, archive)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default='data',  help="Output folder")
    parser.add_argument("--name", type=str, default='Test',  help="Dataset name")
    args = parser.parse_args()
    get_data(os.path.join(args.output, args.name))