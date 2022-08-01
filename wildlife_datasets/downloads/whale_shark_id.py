import os
import argparse
from . import utils

def get_data(root):
    with utils.data_directory(root):
        url = 'https://lilablobssc.blob.core.windows.net/whale-shark-id/whaleshark.coco.tar.gz'
        archive = 'whaleshark.coco.tar.gz'
        utils.download_url(url, archive)
        utils.extract_archive(archive, delete=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default='data',  help="Output folder")
    parser.add_argument("--name", type=str, default='WhaleSharkID',  help="Dataset name")
    args = parser.parse_args()
    get_data(os.path.join(args.output, args.name))