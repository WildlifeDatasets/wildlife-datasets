import os
name = 'ATRW'
urls = [
    # Wild dataset (Detection)
    'https://lilablobssc.blob.core.windows.net/cvwc2019/train/atrw_detection_train.tar.gz',
    'https://lilablobssc.blob.core.windows.net/cvwc2019/train/atrw_anno_detection_train.tar.gz',
    'https://lilablobssc.blob.core.windows.net/cvwc2019/test/atrw_detection_test.tar.gz',

    # Re-ID dataset
    'https://lilablobssc.blob.core.windows.net/cvwc2019/train/atrw_reid_train.tar.gz',
    'https://lilablobssc.blob.core.windows.net/cvwc2019/train/atrw_anno_reid_train.tar.gz',
    'https://lilablobssc.blob.core.windows.net/cvwc2019/test/atrw_reid_test.tar.gz',
    'https://lilablobssc.blob.core.windows.net/cvwc2019/test/atrw_anno_reid_test.tar.gz',
    ]
for url in urls:
    os.system(f"wget -P '../datasets/{name}' {url}")