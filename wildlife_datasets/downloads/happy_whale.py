import os
import argparse
if __name__ == '__main__':
    import utils
else:
    from . import utils

def get_data(root):
    with utils.data_directory(root):
        os.system(f"kaggle competitions download -c happy-whale-and-dolphin")
        utils.extract_archive('happy-whale-and-dolphin.zip', delete=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default='data',  help="Output folder")
    parser.add_argument("--name", type=str, default='HappyWhale',  help="Dataset name")
    args = parser.parse_args()
    get_data(os.path.join(args.output, args.name))