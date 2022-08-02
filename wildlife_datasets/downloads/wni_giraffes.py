import os
import argparse
import shutil
if __name__ == '__main__':
    import utils
else:
    from . import utils


def get_data(root):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    with utils.data_directory(root):
        # TODO: does not work. it is specific for specific distros, isnt it?
        # TODO: Yes this is Ubuntu/linux specific.

        # Copy azcopy from utils to working directory.
        shutil.copy(os.path.join(script_dir, 'utils', 'azcopy'), os.path.join(root, 'azcopy'))

        # Images
        url = "https://lilablobssc.blob.core.windows.net/wni-giraffes/wni_giraffes_train_images.zip"
        archive = 'wni_giraffes_train_images.zip'
        os.system(f'./azcopy cp {url} {archive}')
        utils.extract_archive(archive, delete=True)

        # Metadata
        url = 'https://lilablobssc.blob.core.windows.net/wni-giraffes/wni_giraffes_train.zip'
        archive = 'wni_giraffes_train.zip'
        utils.download_url(url, archive)
        utils.extract_archive(archive, delete=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default='data',  help="Output folder")
    parser.add_argument("--name", type=str, default='WNIGiraffes',  help="Dataset name")
    args = parser.parse_args()
    get_data(os.path.join(args.output, args.name))