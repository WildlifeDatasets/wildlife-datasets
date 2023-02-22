import os
import argparse
import shutil
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
        utils.kaggle_download(f"competitions download -c noaa-right-whale-recognition")

def extract(root):
    utils.print_start2(root)
    with utils.data_directory(root):
        utils.extract_archive('noaa-right-whale-recognition.zip', delete=True)
        utils.extract_archive('imgs.zip', delete=True)

        # Move misplaced image
        shutil.move('w_7489.jpg', 'imgs')
        os.remove('w_7489.jpg.zip')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default='data',  help="Output folder")
    parser.add_argument("--name", type=str, default='NOAARightWhale',  help="Dataset name")
    args = parser.parse_args()
    get_data(os.path.join(args.output, args.name))