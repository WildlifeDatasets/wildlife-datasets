import os
url = 'https://github.com/cvjena/chimpanzee_faces/archive/refs/heads/master.zip'
name = 'CTai'
# data in datasets_cropped_chimpanzee_faces/data_CTai/face_images
# csv file datasets_cropped_chimpanzee_faces/data_CTai/annotations_ctai.txt
# include additional information on age, sex and keypoints

os.system(f"wget -P '../datasets/{name}' {url}")