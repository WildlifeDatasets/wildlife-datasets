import os
import argparse
if __name__ == '__main__':
    import utils
else:
    from . import utils

def get_data(root):
    utils.print_start(root)
    with utils.data_directory(root):
        url = 'https://data.bris.ac.uk/datasets/tar/3owflku95bxsx24643cybxu3qh.zip'
        archive = '3owflku95bxsx24643cybxu3qh.zip'
        utils.download_url(url, archive)
        utils.extract_archive(archive, delete=True)
    utils.print_finish(root)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default='data',  help="Output folder")
    parser.add_argument("--name", type=str, default='AerialCattle2017',  help="Dataset name")
    args = parser.parse_args()
    get_data(os.path.join(args.output, args.name))