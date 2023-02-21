import os
import argparse
if __name__ == '__main__':
    import utils
else:
    from . import utils

def get_data(root):
    utils.print_start(root)
    with utils.data_directory(root):
        os.system(f"kaggle competitions download -c humpback-whale-identification")
        utils.extract_archive('humpback-whale-identification.zip', delete=True)
    utils.print_finish(root)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default='data',  help="Output folder")
    parser.add_argument("--name", type=str, default='HumpbackWhale',  help="Dataset name")
    args = parser.parse_args()
    get_data(os.path.join(args.output, args.name))