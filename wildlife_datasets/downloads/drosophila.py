import os
import argparse
if __name__ == '__main__':
    import utils
else:
    from . import utils

def get_data(root):
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
            utils.extract_archive(archive, extract_path=os.path.splitext(archive)[0], delete=True)

        # Download README
        url = 'https://dataverse.scholarsportal.info/api/access/datafile/71134'
        file = 'ReadMe_Drosophila.pdf'
        utils.download_url(url, file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default='data',  help="Output folder")
    parser.add_argument("--name", type=str, default='Drosophila',  help="Dataset name")
    args = parser.parse_args()
    get_data(os.path.join(args.output, args.name))