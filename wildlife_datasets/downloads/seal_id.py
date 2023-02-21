import os
import argparse
import shutil
if __name__ == '__main__':
    import utils
else:
    from . import utils

def get_data(root, url=''):
    utils.print_start(root)
    if url == '':
        raise(Exception('URL must be provided for SealID.\nCheck https://wildlifedatasets.github.io/wildlife-datasets/downloads/#sealid'))
    with utils.data_directory(root):        
        archive = '22b5191e-f24b-4457-93d3-95797c900fc0_ui65zipk.zip'
        utils.download_url(url, archive)
        utils.extract_archive(archive, delete=True)
        utils.extract_archive(os.path.join('SealID', 'full images.zip'), delete=True)
        utils.extract_archive(os.path.join('SealID', 'patches.zip'), delete=True)
        
        # Create new folder for segmented images
        folder_new = os.getcwd() + 'Segmented'
        if not os.path.exists(folder_new):
            os.makedirs(folder_new)
        
        # Move segmented images to new folder
        folder_move = os.path.join('patches', 'segmented')
        shutil.move(folder_move, os.path.join(folder_new, folder_move))
        folder_move = os.path.join('full images', 'segmented_database')
        shutil.move(folder_move, os.path.join(folder_new, folder_move))
        folder_move = os.path.join('full images', 'segmented_query')
        shutil.move(folder_move, os.path.join(folder_new, folder_move))
        file_copy = os.path.join('patches', 'annotation.csv')
        shutil.copy(file_copy, os.path.join(folder_new, file_copy))
        file_copy = os.path.join('full images', 'annotation.csv')
        shutil.copy(file_copy, os.path.join(folder_new, file_copy))        
    utils.print_finish(root)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default='data',  help="Output folder")
    parser.add_argument("--name", type=str, default='SealID',  help="Dataset name")
    parser.add_argument("--url", type=str, default='',  help="Url for download")
    args = parser.parse_args()
    get_data(os.path.join(args.output, args.name), args.url)