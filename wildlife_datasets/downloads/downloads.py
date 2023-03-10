import os
import shutil
from . import utils

class Downloader():
    def __init__(self, name=None):
        if name is None:
            self.name = self.__class__.__name__
        else:
            self.name = name

    def get_data(self, root, *args, **kwargs):
        self.download_print(os.path.join(root, self.name), *args, **kwargs)
        self.extract_print(os.path.join(root, self.name), *args, **kwargs)
        print('DATASET %s: FINISHED. If mass downloading, you can remove it from the list.' % self.name)
        print('')
    
    def download_print(self, root, *args, **kwargs):
        print('DATASET %s: DOWNLOADING STARTED.' % self.name)
        self.download(root, *args, **kwargs)

    def extract_print(self, root, *args, **kwargs):
        print('DATASET %s: EXTRACTING STARTED.' % self.name)
        self.extract(root, *args, **kwargs)

    def download(self, *args, **kwargs):
        raise NotImplemented('Needs to be implemented by subclasses.')
    
    def extract(self, *args, **kwargs):
        raise NotImplemented('Needs to be implemented by subclasses.')



class AAUZebraFish(Downloader):
    def download(self, root):
        with utils.data_directory(root):
            utils.kaggle_download(f"datasets download -d 'aalborguniversity/aau-zebrafish-reid' --unzip")

    def extract(self, root):
        pass

class AerialCattle2017(Downloader):
    def download(self, root):
        with utils.data_directory(root):
            url = 'https://data.bris.ac.uk/datasets/tar/3owflku95bxsx24643cybxu3qh.zip'
            archive = '3owflku95bxsx24643cybxu3qh.zip'
            utils.download_url(url, archive)

    def extract(self, root):
        with utils.data_directory(root):
            archive = '3owflku95bxsx24643cybxu3qh.zip'
            utils.extract_archive(archive, delete=True)

class ATRW(Downloader):
    def download(self, root):
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

            # Download
            for url, archive in downloads:
                utils.download_url(url, archive)

            # Download evaluation scripts
            url = 'https://github.com/cvwc2019/ATRWEvalScript/archive/refs/heads/main.zip'
            archive = 'main.zip'
            utils.download_url(url, archive)

    def extract(self, root):
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
                archive_name = archive.split('.')[0]
                utils.extract_archive(archive, archive_name, delete=True)

            # Download evaluation scripts
            archive = 'main.zip'
            utils.extract_archive(archive, 'eval_script', delete=True)

class BelugaID(Downloader):
    def download(self, root):
        with utils.data_directory(root):
            url = 'https://lilablobssc.blob.core.windows.net/liladata/wild-me/beluga.coco.tar.gz'
            archive = 'beluga.coco.tar.gz'
            utils.download_url(url, archive)

    def extract(self, root):
        with utils.data_directory(root):
            archive = 'beluga.coco.tar.gz'
            utils.extract_archive(archive, delete=True)

class BirdIndividualID(Downloader):
    def download(self, root):
        with utils.data_directory(root):
            # TODO: why not the automatic download?
            # Try automatic download
            import gdown
            url = 'https://drive.google.com/uc?id=1YT4w8yF44D-y9kdzgF38z2uYbHfpiDOA'
            archive = 'ferreira_et_al_2020.zip'
            #gdown.download(
            #    f"https://drive.google.com/uc?export=download&confirm=pbef&id=1YT4w8yF44D-y9kdzgF38z2uYbHfpiDOA",
            #    'qwe.zip'
            #)
            #qwqjwdoijqwo
            gdown.download(url, archive, quiet=False)
            #utils.extract_archive(archive, delete=True)

            # Download manually from:
            url = 'https://drive.google.com/drive/folders/1YkH_2DNVBOKMNGxDinJb97y2T8_wRTZz'

            # Upload to kaggle and download
            utils.kaggle_download(f"datasets download -d 'vojtacermak/birds' --unzip")

    def extract(self, root):
        with utils.data_directory(root):
            # Create new folder for segmented images
            folder_new = os.getcwd() + 'Segmented'
            if not os.path.exists(folder_new):
                os.makedirs(folder_new)

            # Move segmented images to new folder
            folder_move = 'Cropped_pictures'
            shutil.move(folder_move, os.path.join(folder_new, folder_move))

class BristolGorillas2020(Downloader):
    def download(self, root):
        with utils.data_directory(root):
            # TODO: does not work
            url = 'https://data.bris.ac.uk/datasets/tar/jf0859kboy8k2ufv60dqeb2t8.zip'
            archive = 'jf0859kboy8k2ufv60dqeb2t8.zip'
            utils.download_url(url, archive)

    def extract(self, root):
        with utils.data_directory(root):
            archive = 'jf0859kboy8k2ufv60dqeb2t8.zip'
            utils.extract_archive(archive, delete=True)

class CTai(Downloader):
    def download(self, root):
        with utils.data_directory(root):
            url = 'https://github.com/cvjena/chimpanzee_faces/archive/refs/heads/master.zip'
            archive = 'master.zip'
            utils.download_url(url, archive)

    def extract(self, root):
        with utils.data_directory(root):
            archive = 'master.zip'
            utils.extract_archive(archive, delete=True)

            # Cleanup
            shutil.rmtree('chimpanzee_faces-master/datasets_cropped_chimpanzee_faces/data_CZoo')

class CZoo(Downloader):
    def download(self, root):
        with utils.data_directory(root):
            url = 'https://github.com/cvjena/chimpanzee_faces/archive/refs/heads/master.zip'
            archive = 'master.zip'
            utils.download_url(url, archive)

    def extract(self, root):
        with utils.data_directory(root):
            archive = 'master.zip'
            utils.extract_archive(archive, delete=True)

            # Cleanup
            shutil.rmtree('chimpanzee_faces-master/datasets_cropped_chimpanzee_faces/data_CTai')

class Cows2021(Downloader):
    def download(self, root):
        with utils.data_directory(root):
            url = 'https://data.bris.ac.uk/datasets/tar/4vnrca7qw1642qlwxjadp87h7.zip'
            archive = '4vnrca7qw1642qlwxjadp87h7.zip'
            utils.download_url(url, archive)

    def extract(self, root):
        with utils.data_directory(root):
            archive = '4vnrca7qw1642qlwxjadp87h7.zip'
            utils.extract_archive(archive, delete=True)

class Drosophila(Downloader):
    def download(self, root):
        with utils.data_directory(root):
            downloads = [
                ('https://dataverse.scholarsportal.info/api/access/datafile/71066', 'week1_Day1_train_01to05.zip'),
                ('https://dataverse.scholarsportal.info/api/access/datafile/71067', 'week1_Day1_train_06to10.zip'),
                ('https://dataverse.scholarsportal.info/api/access/datafile/71068', 'week1_Day1_train_11to15.zip'),
                ('https://dataverse.scholarsportal.info/api/access/datafile/71069', 'week1_Day1_train_16to20.zip'),
                ('https://dataverse.scholarsportal.info/api/access/datafile/71065', 'week1_Day1_val.zip'),
                ('https://dataverse.scholarsportal.info/api/access/datafile/71071', 'week1_Day2_train_01to05.zip'),
                ('https://dataverse.scholarsportal.info/api/access/datafile/71072', 'week1_Day2_train_06to10.zip'),
                ('https://dataverse.scholarsportal.info/api/access/datafile/71073', 'week1_Day2_train_11to15.zip'),
                ('https://dataverse.scholarsportal.info/api/access/datafile/71075', 'week1_Day2_train_16to20.zip'),
                ('https://dataverse.scholarsportal.info/api/access/datafile/71070', 'week1_Day2_val.zip'),
                ('https://dataverse.scholarsportal.info/api/access/datafile/71077', 'week1_Day3_01to04.zip'),
                ('https://dataverse.scholarsportal.info/api/access/datafile/71078', 'week1_Day3_05to08.zip'),
                ('https://dataverse.scholarsportal.info/api/access/datafile/71079', 'week1_Day3_09to12.zip'),
                ('https://dataverse.scholarsportal.info/api/access/datafile/71080', 'week1_Day3_13to16.zip'),
                ('https://dataverse.scholarsportal.info/api/access/datafile/71081', 'week1_Day3_17to20.zip'),
                ('https://dataverse.scholarsportal.info/api/access/datafile/71083', 'week2_Day1_train_01to05.zip'),
                ('https://dataverse.scholarsportal.info/api/access/datafile/71084', 'week2_Day1_train_06to10.zip'),
                ('https://dataverse.scholarsportal.info/api/access/datafile/71085', 'week2_Day1_train_11to15.zip'),
                ('https://dataverse.scholarsportal.info/api/access/datafile/71086', 'week2_Day1_train_16to20.zip'),        
                ('https://dataverse.scholarsportal.info/api/access/datafile/71082', 'week2_Day1_val.zip'),
                ('https://dataverse.scholarsportal.info/api/access/datafile/71094', 'week2_Day2_train_01to05.zip'),
                ('https://dataverse.scholarsportal.info/api/access/datafile/71095', 'week2_Day2_train_06to10.zip'),
                ('https://dataverse.scholarsportal.info/api/access/datafile/71109', 'week2_Day2_train_11to15.zip'),
                ('https://dataverse.scholarsportal.info/api/access/datafile/71110', 'week2_Day2_train_16to20.zip'),
                ('https://dataverse.scholarsportal.info/api/access/datafile/71093', 'week2_Day2_val.zip'),
                ('https://dataverse.scholarsportal.info/api/access/datafile/71111', 'week2_Day3_01to04.zip'),
                ('https://dataverse.scholarsportal.info/api/access/datafile/71112', 'week2_Day3_05to08.zip'),
                ('https://dataverse.scholarsportal.info/api/access/datafile/71115', 'week2_Day3_09to12.zip'),
                ('https://dataverse.scholarsportal.info/api/access/datafile/71117', 'week2_Day3_13to16.zip'),
                ('https://dataverse.scholarsportal.info/api/access/datafile/71118', 'week2_Day3_17to20.zip'),
                ('https://dataverse.scholarsportal.info/api/access/datafile/71119', 'week3_Day1_train_01to05.zip'),
                ('https://dataverse.scholarsportal.info/api/access/datafile/71120', 'week3_Day1_train_06to10.zip'),
                ('https://dataverse.scholarsportal.info/api/access/datafile/71121', 'week3_Day1_train_11to15.zip'),
                ('https://dataverse.scholarsportal.info/api/access/datafile/71124', 'week3_Day1_train_16to20.zip'),
                ('https://dataverse.scholarsportal.info/api/access/datafile/71097', 'week3_Day1_val.zip'),
                ('https://dataverse.scholarsportal.info/api/access/datafile/71125', 'week3_Day2_train_01to05.zip'),
                ('https://dataverse.scholarsportal.info/api/access/datafile/71126', 'week3_Day2_train_06to10.zip'),
                ('https://dataverse.scholarsportal.info/api/access/datafile/71127', 'week3_Day2_train_11to15.zip'),
                ('https://dataverse.scholarsportal.info/api/access/datafile/71128', 'week3_Day2_train_16to20.zip'),
                ('https://dataverse.scholarsportal.info/api/access/datafile/71107', 'week3_Day2_val.zip'),
                ('https://dataverse.scholarsportal.info/api/access/datafile/71129', 'week3_Day3_01to04.zip'),
                ('https://dataverse.scholarsportal.info/api/access/datafile/71130', 'week3_Day3_05to08.zip'),
                ('https://dataverse.scholarsportal.info/api/access/datafile/71131', 'week3_Day3_09to12.zip'),
                ('https://dataverse.scholarsportal.info/api/access/datafile/71132', 'week3_Day3_13to16.zip'),
                ('https://dataverse.scholarsportal.info/api/access/datafile/71133', 'week3_Day3_17to20.zip'),
            ]

            for url, archive in downloads:
                utils.download_url(url, archive)

            # Download README
            url = 'https://dataverse.scholarsportal.info/api/access/datafile/71134'
            file = 'ReadMe_Drosophila.pdf'
            utils.download_url(url, file)

    def extract(self, root):
        with utils.data_directory(root):
            downloads = [
                ('https://dataverse.scholarsportal.info/api/access/datafile/71066', 'week1_Day1_train_01to05.zip'),
                ('https://dataverse.scholarsportal.info/api/access/datafile/71067', 'week1_Day1_train_06to10.zip'),
                ('https://dataverse.scholarsportal.info/api/access/datafile/71068', 'week1_Day1_train_11to15.zip'),
                ('https://dataverse.scholarsportal.info/api/access/datafile/71069', 'week1_Day1_train_16to20.zip'),
                ('https://dataverse.scholarsportal.info/api/access/datafile/71065', 'week1_Day1_val.zip'),
                ('https://dataverse.scholarsportal.info/api/access/datafile/71071', 'week1_Day2_train_01to05.zip'),
                ('https://dataverse.scholarsportal.info/api/access/datafile/71072', 'week1_Day2_train_06to10.zip'),
                ('https://dataverse.scholarsportal.info/api/access/datafile/71073', 'week1_Day2_train_11to15.zip'),
                ('https://dataverse.scholarsportal.info/api/access/datafile/71075', 'week1_Day2_train_16to20.zip'),
                ('https://dataverse.scholarsportal.info/api/access/datafile/71070', 'week1_Day2_val.zip'),
                ('https://dataverse.scholarsportal.info/api/access/datafile/71077', 'week1_Day3_01to04.zip'),
                ('https://dataverse.scholarsportal.info/api/access/datafile/71078', 'week1_Day3_05to08.zip'),
                ('https://dataverse.scholarsportal.info/api/access/datafile/71079', 'week1_Day3_09to12.zip'),
                ('https://dataverse.scholarsportal.info/api/access/datafile/71080', 'week1_Day3_13to16.zip'),
                ('https://dataverse.scholarsportal.info/api/access/datafile/71081', 'week1_Day3_17to20.zip'),
                ('https://dataverse.scholarsportal.info/api/access/datafile/71083', 'week2_Day1_train_01to05.zip'),
                ('https://dataverse.scholarsportal.info/api/access/datafile/71084', 'week2_Day1_train_06to10.zip'),
                ('https://dataverse.scholarsportal.info/api/access/datafile/71085', 'week2_Day1_train_11to15.zip'),
                ('https://dataverse.scholarsportal.info/api/access/datafile/71086', 'week2_Day1_train_16to20.zip'),        
                ('https://dataverse.scholarsportal.info/api/access/datafile/71082', 'week2_Day1_val.zip'),
                ('https://dataverse.scholarsportal.info/api/access/datafile/71094', 'week2_Day2_train_01to05.zip'),
                ('https://dataverse.scholarsportal.info/api/access/datafile/71095', 'week2_Day2_train_06to10.zip'),
                ('https://dataverse.scholarsportal.info/api/access/datafile/71109', 'week2_Day2_train_11to15.zip'),
                ('https://dataverse.scholarsportal.info/api/access/datafile/71110', 'week2_Day2_train_16to20.zip'),
                ('https://dataverse.scholarsportal.info/api/access/datafile/71093', 'week2_Day2_val.zip'),
                ('https://dataverse.scholarsportal.info/api/access/datafile/71111', 'week2_Day3_01to04.zip'),
                ('https://dataverse.scholarsportal.info/api/access/datafile/71112', 'week2_Day3_05to08.zip'),
                ('https://dataverse.scholarsportal.info/api/access/datafile/71115', 'week2_Day3_09to12.zip'),
                ('https://dataverse.scholarsportal.info/api/access/datafile/71117', 'week2_Day3_13to16.zip'),
                ('https://dataverse.scholarsportal.info/api/access/datafile/71118', 'week2_Day3_17to20.zip'),
                ('https://dataverse.scholarsportal.info/api/access/datafile/71119', 'week3_Day1_train_01to05.zip'),
                ('https://dataverse.scholarsportal.info/api/access/datafile/71120', 'week3_Day1_train_06to10.zip'),
                ('https://dataverse.scholarsportal.info/api/access/datafile/71121', 'week3_Day1_train_11to15.zip'),
                ('https://dataverse.scholarsportal.info/api/access/datafile/71124', 'week3_Day1_train_16to20.zip'),
                ('https://dataverse.scholarsportal.info/api/access/datafile/71097', 'week3_Day1_val.zip'),
                ('https://dataverse.scholarsportal.info/api/access/datafile/71125', 'week3_Day2_train_01to05.zip'),
                ('https://dataverse.scholarsportal.info/api/access/datafile/71126', 'week3_Day2_train_06to10.zip'),
                ('https://dataverse.scholarsportal.info/api/access/datafile/71127', 'week3_Day2_train_11to15.zip'),
                ('https://dataverse.scholarsportal.info/api/access/datafile/71128', 'week3_Day2_train_16to20.zip'),
                ('https://dataverse.scholarsportal.info/api/access/datafile/71107', 'week3_Day2_val.zip'),
                ('https://dataverse.scholarsportal.info/api/access/datafile/71129', 'week3_Day3_01to04.zip'),
                ('https://dataverse.scholarsportal.info/api/access/datafile/71130', 'week3_Day3_05to08.zip'),
                ('https://dataverse.scholarsportal.info/api/access/datafile/71131', 'week3_Day3_09to12.zip'),
                ('https://dataverse.scholarsportal.info/api/access/datafile/71132', 'week3_Day3_13to16.zip'),
                ('https://dataverse.scholarsportal.info/api/access/datafile/71133', 'week3_Day3_17to20.zip'),
            ]

            for url, archive in downloads:
                utils.extract_archive(archive, extract_path=os.path.splitext(archive)[0], delete=True)

class FriesianCattle2015(Downloader):
    def download(self, root):
        with utils.data_directory(root):
            url = 'https://data.bris.ac.uk/datasets/wurzq71kfm561ljahbwjhx9n3/wurzq71kfm561ljahbwjhx9n3.zip'
            archive = 'wurzq71kfm561ljahbwjhx9n3.zip'
            utils.download_url(url, archive)

    def extract(self, root):
        with utils.data_directory(root):
            archive = 'wurzq71kfm561ljahbwjhx9n3.zip'
            utils.extract_archive(archive, delete=True)

class FriesianCattle2017(Downloader):
    def download(self, root):
        with utils.data_directory(root):
            url = 'https://data.bris.ac.uk/datasets/2yizcfbkuv4352pzc32n54371r/2yizcfbkuv4352pzc32n54371r.zip'
            archive = '2yizcfbkuv4352pzc32n54371r.zip'
            utils.download_url(url, archive)

    def extract(self, root):
        with utils.data_directory(root):
            archive = '2yizcfbkuv4352pzc32n54371r.zip'
            utils.extract_archive(archive, delete=True)

class GiraffeZebraID(Downloader):
    def download(self, root):
        with utils.data_directory(root):
            url = 'https://lilablobssc.blob.core.windows.net/giraffe-zebra-id/gzgc.coco.tar.gz'
            archive = 'gzgc.coco.tar.gz'
            utils.download_url(url, archive)

    def extract(self, root):
        with utils.data_directory(root):
            archive = 'gzgc.coco.tar.gz'
            utils.extract_archive(archive, delete=True)

class Giraffes(Downloader):
    def download(self, root):
        with utils.data_directory(root):
            url = 'ftp://pbil.univ-lyon1.fr/pub/datasets/miele2021/'
            os.system(f"wget -rpk -l 10 -np -c --random-wait -U Mozilla {url} -P '.' ")

    def extract(self, root):
        pass

class HappyWhale(Downloader):
    def download(self, root):
        with utils.data_directory(root):
            utils.kaggle_download(f"competitions download -c happy-whale-and-dolphin")

    def extract(self, root):
        with utils.data_directory(root):
            utils.extract_archive('happy-whale-and-dolphin.zip', delete=True)

class HumpbackWhaleID(Downloader):
    def download(self, root):
        with utils.data_directory(root):
            utils.kaggle_download(f"competitions download -c humpback-whale-identification")

    def extract(self, root):
        with utils.data_directory(root):
            utils.extract_archive('humpback-whale-identification.zip', delete=True)

class HyenaID2022(Downloader):
    def download(self, root):
        with utils.data_directory(root):
            url = 'https://lilablobssc.blob.core.windows.net/liladata/wild-me/hyena.coco.tar.gz'
            archive = 'hyena.coco.tar.gz'
            utils.download_url(url, archive)

    def extract(self, root):
        with utils.data_directory(root):
            archive = 'hyena.coco.tar.gz'
            utils.extract_archive(archive, delete=True)

class IPanda50(Downloader):
    def download(self, root):
        import gdown
        with utils.data_directory(root):
            downloads = [
                ('https://drive.google.com/uc?id=1nkh-g6a8JvWy-XsMaZqrN2AXoPlaXuFg', 'iPanda50-images.zip'),
                ('https://drive.google.com/uc?id=1gVREtFWkNec4xwqOyKkpuIQIyWU_Y_Ob', 'iPanda50-split.zip'),
                ('https://drive.google.com/uc?id=1jdACN98uOxedZDT-6X3rpbooLAAUEbNY', 'iPanda50-eyes-labels.zip'),
            ]

            for url, archive in downloads:
                gdown.download(url, archive, quiet=False)

    def extract(self, root):
        with utils.data_directory(root):
            downloads = [
                ('https://drive.google.com/uc?id=1nkh-g6a8JvWy-XsMaZqrN2AXoPlaXuFg', 'iPanda50-images.zip'),
                ('https://drive.google.com/uc?id=1gVREtFWkNec4xwqOyKkpuIQIyWU_Y_Ob', 'iPanda50-split.zip'),
                ('https://drive.google.com/uc?id=1jdACN98uOxedZDT-6X3rpbooLAAUEbNY', 'iPanda50-eyes-labels.zip'),
            ]

            for url, archive in downloads:
                utils.extract_archive(archive, delete=True)

class LeopardID2022(Downloader):
    def download(self, root):
        with utils.data_directory(root):
            url = 'https://lilablobssc.blob.core.windows.net/liladata/wild-me/leopard.coco.tar.gz'
            archive = 'leopard.coco.tar.gz'
            utils.download_url(url, archive)

    def extract(self, root):
        with utils.data_directory(root):
            archive = 'leopard.coco.tar.gz'
            utils.extract_archive(archive, delete=True)

class LionData(Downloader):
    def download(self, root):
        with utils.data_directory(root):
            url = 'https://github.com/tvanzyl/wildlife_reidentification/archive/refs/heads/main.zip'
            archive = 'main.zip'
            utils.download_url(url, archive)

    def extract(self, root):
        with utils.data_directory(root):
            archive = 'main.zip'
            utils.extract_archive(archive, delete=True)

            # Cleanup
            shutil.rmtree('wildlife_reidentification-main/Nyala_Data_Zero')

class MacaqueFaces(Downloader):
    def download(self, root):
        with utils.data_directory(root):
            downloads = [
                ('https://github.com/clwitham/MacaqueFaces/raw/master/ModelSet/MacaqueFaces.zip', 'MacaqueFaces.zip'),
                ('https://github.com/clwitham/MacaqueFaces/raw/master/ModelSet/MacaqueFaces_ImageInfo.csv', 'MacaqueFaces_ImageInfo.csv'),
            ]
            for url, file in downloads:
                utils.download_url(url, file)

    def extract(self, root):
        with utils.data_directory(root):
            utils.extract_archive('MacaqueFaces.zip', delete=True)

class NDD20(Downloader):
    def download(self, root):
        with utils.data_directory(root):
            url = 'https://data.ncl.ac.uk/ndownloader/files/22774175'
            archive = 'NDD20.zip'
            utils.download_url(url, archive)

    def extract(self, root):
        with utils.data_directory(root):
            archive = 'NDD20.zip'
            utils.extract_archive(archive, delete=True)    

class NOAARightWhale(Downloader):
    def download(self, root):
        with utils.data_directory(root):
            utils.kaggle_download(f"competitions download -c noaa-right-whale-recognition")

    def extract(self, root):
        with utils.data_directory(root):
            utils.extract_archive('noaa-right-whale-recognition.zip', delete=True)
            utils.extract_archive('imgs.zip', delete=True)

            # Move misplaced image
            shutil.move('w_7489.jpg', 'imgs')
            os.remove('w_7489.jpg.zip')

class NyalaData(Downloader):
    def download(self, root):
        with utils.data_directory(root):
            url = 'https://github.com/tvanzyl/wildlife_reidentification/archive/refs/heads/main.zip'
            archive = 'main.zip'
            utils.download_url(url, archive)

    def extract(self, root):
        with utils.data_directory(root):        
            archive = 'main.zip'
            utils.extract_archive(archive, delete=True)

            # Cleanup
            shutil.rmtree('wildlife_reidentification-main/Lion_Data_Zero')

class OpenCows2020(Downloader):
    def download(self, root):
        with utils.data_directory(root):
            url = 'https://data.bris.ac.uk/datasets/tar/10m32xl88x2b61zlkkgz3fml17.zip'
            archive = '10m32xl88x2b61zlkkgz3fml17.zip'
            utils.download_url(url, archive)

    def extract(self, root):
        with utils.data_directory(root):
            archive = '10m32xl88x2b61zlkkgz3fml17.zip'
            utils.extract_archive(archive, delete=True)

class SeaTurtleIDHeads(Downloader):
    def download(self, root):
        with utils.data_directory(root):
            utils.kaggle_download(f"datasets download -d 'wildlifedatasets/seaturtleidheads' --unzip")

    def extract(self, root):
        pass

class SeaTurtleID(Downloader):
    def download(self, root):
        with utils.data_directory(root):
            utils.kaggle_download(f"datasets download -d 'wildlifedatasets/seaturtleid' --unzip")

    def extract(self, root):
        pass

class SealID(Downloader):
    def download(self, root, url):
        if url == '':
            raise(Exception('URL must be provided for SealID.\nCheck https://wildlifedatasets.github.io/wildlife-datasets/downloads/#sealid'))
        with utils.data_directory(root):        
            archive = '22b5191e-f24b-4457-93d3-95797c900fc0_ui65zipk.zip'
            utils.download_url(url, archive)

    def extract(self, root):
        with utils.data_directory(root):
            archive = '22b5191e-f24b-4457-93d3-95797c900fc0_ui65zipk.zip'
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

class SMALST(Downloader):
    def download(self, root):
        import gdown
        with utils.data_directory(root):
            url = 'https://drive.google.com/uc?id=1yVy4--M4CNfE5x9wUr1QBmAXEcWb6PWF'
            archive = 'zebra_training_set.zip'
            gdown.download(url, archive, quiet=False)

    def extract(self, root):
        with utils.data_directory(root):
            os.system('jar xvf zebra_training_set.zip')
            os.remove('zebra_training_set.zip')
            shutil.rmtree(os.path.join('zebra_training_set', 'annotations'))
            shutil.rmtree(os.path.join('zebra_training_set', 'texmap'))
            shutil.rmtree(os.path.join('zebra_training_set', 'uvflow'))

class StripeSpotter(Downloader):
    def download(self, root):
        with utils.data_directory(root):
            # Download
            urls = [
                'https://storage.googleapis.com/google-code-archive-downloads/v2/code.google.com/stripespotter/data-20110718.zip',
                'https://storage.googleapis.com/google-code-archive-downloads/v2/code.google.com/stripespotter/data-20110718.z02',
                'https://storage.googleapis.com/google-code-archive-downloads/v2/code.google.com/stripespotter/data-20110718.z01',
                ]
            for url in urls:
                os.system(f"wget -P '.' {url}")

    def extract(self, root):
        with utils.data_directory(root):
            # Extract
            os.system(f"zip -s- data-20110718.zip -O data-full.zip")
            os.system(f"unzip data-full.zip")

            # Cleanup
            os.remove('data-20110718.zip')
            os.remove('data-20110718.z01')
            os.remove('data-20110718.z02')
            os.remove('data-full.zip')

class WhaleSharkID(Downloader):
    def download(self, root):
        with utils.data_directory(root):
            url = 'https://lilablobssc.blob.core.windows.net/whale-shark-id/whaleshark.coco.tar.gz'
            archive = 'whaleshark.coco.tar.gz'
            utils.download_url(url, archive)

    def extract(self, root):
        with utils.data_directory(root):
            archive = 'whaleshark.coco.tar.gz'
            utils.extract_archive(archive, delete=True)

class WNIGiraffes(Downloader):
    def download(self, root):
        with utils.data_directory(root):
            # TODO: does not work. it is specific for specific distros, isnt it?
            # TODO: Yes this is Ubuntu/linux specific.
            # TODO: this requires azcopy
            # Copy azcopy from utils to working directory.

            # Images
            url = "https://lilablobssc.blob.core.windows.net/wni-giraffes/wni_giraffes_train_images.zip"
            archive = 'wni_giraffes_train_images.zip'
            os.system(f'azcopy cp {url} {archive}')

            # Metadata
            url = 'https://lilablobssc.blob.core.windows.net/wni-giraffes/wni_giraffes_train.zip'
            archive = 'wni_giraffes_train.zip'
            utils.download_url(url, archive)

    def extract(self, root):
        with utils.data_directory(root):
            # Images
            archive = 'wni_giraffes_train_images.zip'
            utils.extract_archive(archive, delete=True)

            # Metadata
            archive = 'wni_giraffes_train.zip'
            utils.extract_archive(archive, delete=True)

class ZindiTurtleRecall(Downloader):
    def download(self, root):
        with utils.data_directory(root):
            downloads = [
                ('https://storage.googleapis.com/dm-turtle-recall/train.csv', 'train.csv'),
                ('https://storage.googleapis.com/dm-turtle-recall/extra_images.csv', 'extra_images.csv'),
                ('https://storage.googleapis.com/dm-turtle-recall/test.csv', 'test.csv'),
                ('https://storage.googleapis.com/dm-turtle-recall/images.tar', 'images.tar'),
            ]
            for url, file in downloads:
                utils.download_url(url, file)

    def extract(self, root):
        with utils.data_directory(root):
            utils.extract_archive('images.tar', 'images', delete=True)


class Segmented(Downloader):
    def get_data(self, root):
        with utils.data_directory(root):
            print("You are trying to download a segmented dataset %s. \
                    It is already included in its non-segmented version download. \
                    Skipping." % os.path.split(root)[1])

