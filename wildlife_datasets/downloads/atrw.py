import os
import argparse
if __name__ == '__main__':
    import utils
else:
    from . import utils

def get_data(root):
    with utils.data_directory(root):
        downloads = [
            # Wild dataset (Detection)
            ('https://lilablobssc.blob.core.windows.net/cvwc2019/test/atrw_detection_test.tar.gz', 'atrw_detection_test.tar.gz'),

            # Re-ID dataset
            ('https://lilablobssc.blob.core.windows.net/cvwc2019/train/atrw_reid_train.tar.gz', 'atrw_reid_train.tar.gz'),
            ('https://lilablobssc.blob.core.windows.net/cvwc2019/train/atrw_anno_reid_train.tar.gz', 'atrw_anno_reid_train.tar.gz'),
            ('https://lilablobssc.blob.core.windows.net/cvwc2019/test/atrw_reid_test.tar.gz', 'atrw_reid_test.tar.gz'),
            ('https://lilablobssc.blob.core.windows.net/cvwc2019/test/atrw_anno_reid_test.tar.gz', 'atrw_anno_reid_test.tar.gz'),
            ]

        # Download and extract
        for url, archive in downloads:
            utils.download_url(url, archive)
            archive_name = archive.split('.')[0]
            utils.extract_archive(archive, archive_name, delete=True)

        # Download evaluation scripts
        url = 'https://github.com/cvwc2019/ATRWEvalScript/archive/refs/heads/main.zip'
        archive = 'main.zip'
        utils.download_url(url, archive)
        utils.extract_archive(archive, 'eval_script', delete=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default='data',  help="Output folder")
    parser.add_argument("--name", type=str, default='ATRW',  help="Dataset name")
    args = parser.parse_args()
    get_data(os.path.join(args.output, args.name))