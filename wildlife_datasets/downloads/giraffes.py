import os
import argparse
if __name__ == '__main__':
    import utils
else:
    from . import utils

def get_data(root):
    download(root)
    extract(root)
    utils.print_finish(root)

def download(root):
    utils.print_start1(root)
    with utils.data_directory(root):
        url = 'ftp://pbil.univ-lyon1.fr/pub/datasets/miele2021/'
        os.system(f"wget -rpk -l 10 -np -c --random-wait -U Mozilla {url} -P '.' ")

def extract(root):
    utils.print_start2(root)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default='data',  help="Output folder")
    parser.add_argument("--name", type=str, default='Giraffes',  help="Dataset name")
    args = parser.parse_args()
    get_data(os.path.join(args.output, args.name))