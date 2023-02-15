import os
import argparse
import shutil
if __name__ == '__main__':
    import utils
else:
    from . import utils

def get_data(root):
    with utils.data_directory(root):
        # TODO: why not the automatic download?
        # Try automatic download
        #import gdown
        #url = 'https://drive.google.com/uc?id=1YT4w8yF44D-y9kdzgF38z2uYbHfpiDOA'
        #archive = 'ferreira_et_al_2020.zip'
        #gdown.download(url, archive, quiet=False)
        #utils.extract_archive(archive, delete=True)

        # Download manually from:
        url = 'https://drive.google.com/drive/folders/1YkH_2DNVBOKMNGxDinJb97y2T8_wRTZz'

        # Upload to kaggle and download
        os.system(f"kaggle datasets download -d 'vojtacermak/birds' --unzip")

        # Create new folder for segmented images
        folder_new = os.getcwd() + 'Segmented'
        if not os.path.exists(folder_new):
            os.makedirs(folder_new)

        # Move segmented images to new folder
        folder_move = 'Cropped_pictures'
        shutil.move(folder_move, os.path.join(folder_new, folder_move))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default='data',  help="Output folder")
    parser.add_argument("--name", type=str, default='BirdIndividualID',  help="Dataset name")
    args = parser.parse_args()
    get_data(os.path.join(args.output, args.name))