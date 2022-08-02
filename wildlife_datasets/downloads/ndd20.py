import os
import argparse
if __name__ == '__main__':
    import utils
else:
    from . import utils

def get_data(root):
    with utils.data_directory(root):
        url = 'https://data.ncl.ac.uk/ndownloader/files/22774175'
        archive = 'NDD20.zip'
        utils.download_url(url, archive)
        utils.extract_archive(archive, delete=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default='data',  help="Output folder")
    parser.add_argument("--name", type=str, default='NDD20',  help="Dataset name")
    args = parser.parse_args()
    get_data(os.path.join(args.output, args.name))