import os
import argparse
if __name__ == '__main__':
    import utils
else:
    from . import utils

def get_data(root):
    import gdown
    with utils.data_directory(root):
        downloads = [
            ('https://drive.google.com/uc?id=1nkh-g6a8JvWy-XsMaZqrN2AXoPlaXuFg', 'iPanda50-images.zip'),
            ('https://drive.google.com/uc?id=1gVREtFWkNec4xwqOyKkpuIQIyWU_Y_Ob', 'iPanda50-split.zip'),
            ('https://drive.google.com/uc?id=1jdACN98uOxedZDT-6X3rpbooLAAUEbNY', 'iPanda50-eyes-labels.zip'),
        ]

        for url, archive in downloads:
            gdown.download(url, archive, quiet=False)
            utils.extract_archive(archive, delete=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default='data',  help="Output folder")
    parser.add_argument("--name", type=str, default='IPanda50',  help="Dataset name")
    args = parser.parse_args()
    get_data(os.path.join(args.output, args.name))