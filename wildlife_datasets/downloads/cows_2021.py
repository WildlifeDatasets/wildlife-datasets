import os
import argparse
if __name__ == '__main__':
    import utils
else:
    from . import utils

def get_data(root):
    with utils.data_directory(root):
        url = 'https://data.bris.ac.uk/datasets/tar/4vnrca7qw1642qlwxjadp87h7.zip'
        archive = '4vnrca7qw1642qlwxjadp87h7.zip'
        utils.download_url(url, archive)
        utils.extract_archive(archive, delete=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default='data',  help="Output folder")
    parser.add_argument("--name", type=str, default='Cows2021',  help="Dataset name")
    args = parser.parse_args()
    get_data(os.path.join(args.output, args.name))