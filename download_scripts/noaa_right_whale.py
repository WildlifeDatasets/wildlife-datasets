import os
import argparse
import utils
import shutil

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=str, default='../datasets',  help="Output folder")
parser.add_argument("--name", type=str, default='NOAARightWhale',  help="Dataset name")
args = parser.parse_args()

directory = os.path.join(args.output, args.name)
if not os.path.exists(directory):
    os.makedirs(directory)
os.chdir(directory)

# Download and extract
os.system(f"kaggle competitions download -c noaa-right-whale-recognition")
utils.extract_archive('noaa-right-whale-recognition.zip', delete=True)
utils.extract_archive('imgs.zip', delete=True)

# Move misplaced image
shutil.move('w_7489.jpg', 'imgs')
os.remove('w_7489.jpg.zip')