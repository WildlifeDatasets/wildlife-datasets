import os
import argparse
import utils

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=str, default='../datasets',  help="Output folder")
parser.add_argument("--name", type=str, default='ATRW',  help="Dataset name")
args = parser.parse_args()

directory = os.path.join(args.output, args.name)
if not os.path.exists(directory):
    os.makedirs(directory)
os.chdir(directory)

downloads = [
    # Wild dataset (Detection)
    ('https://lilablobssc.blob.core.windows.net/cvwc2019/train/atrw_detection_train.tar.gz', 'atrw_detection_train.tar.gz'),
    ('https://lilablobssc.blob.core.windows.net/cvwc2019/train/atrw_anno_detection_train.tar.gz', 'atrw_anno_detection_train.tar.gz'),
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