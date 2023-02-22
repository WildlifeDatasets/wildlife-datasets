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
        # Download
        urls = [
            'https://storage.googleapis.com/google-code-archive-downloads/v2/code.google.com/stripespotter/data-20110718.zip',
            'https://storage.googleapis.com/google-code-archive-downloads/v2/code.google.com/stripespotter/data-20110718.z02',
            'https://storage.googleapis.com/google-code-archive-downloads/v2/code.google.com/stripespotter/data-20110718.z01',
            ]
        for url in urls:
            os.system(f"wget -P '.' {url}")

def extract(root):
    utils.print_start2(root)
    with utils.data_directory(root):
        # Extract
        os.system(f"zip -s- data-20110718.zip -O data-full.zip")
        os.system(f"unzip data-full.zip")

        # Cleanup
        os.remove('data-20110718.zip')
        os.remove('data-20110718.z01')
        os.remove('data-20110718.z02')
        os.remove('data-full.zip')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default='data',  help="Output folder")
    parser.add_argument("--name", type=str, default='StripeSpotter',  help="Dataset name")
    args = parser.parse_args()
    get_data(os.path.join(args.output, args.name))