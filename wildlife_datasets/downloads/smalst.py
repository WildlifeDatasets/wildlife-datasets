import os
import gdown
import shutil
import argparse
if __name__ == '__main__':
    import utils
else:
    from . import utils

def get_data(root):
    with utils.data_directory(root):
        url = 'https://drive.google.com/uc?id=1yVy4--M4CNfE5x9wUr1QBmAXEcWb6PWF'
        archive = 'zebra_training_set.zip'
        gdown.download(url, archive, quiet=False)

        os.system('jar xvf zebra_training_set.zip')
        os.remove('zebra_training_set.zip')
        shutil.rmtree(os.path.join('zebra_training_set', 'annotations'))
        shutil.rmtree(os.path.join('zebra_training_set', 'texmap'))
        shutil.rmtree(os.path.join('zebra_training_set', 'uvflow'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default='data',  help="Output folder")
    parser.add_argument("--name", type=str, default='SMALST',  help="Dataset name")
    args = parser.parse_args()
    get_data(os.path.join(args.output, args.name))