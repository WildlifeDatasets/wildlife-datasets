import os
url = 'https://github.com/cvjena/chimpanzee_faces/archive/refs/heads/master.zip'
name = 'CZoo'
# data in datasets_cropped_chimpanzee_faces/data_CZoo/face_images
# csv file datasets_cropped_chimpanzee_faces/data_CZoo/annotations_czoo.txt
# include additional information on age, sex and keypoints

os.system(f"wget -P '../datasets/{name}' {url}")