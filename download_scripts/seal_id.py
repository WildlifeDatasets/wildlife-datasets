

import os
import argparse
import utils

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=str, default='../datasets',  help="Output folder")
parser.add_argument("--name", type=str, default='SealID',  help="Dataset name")
args = parser.parse_args()

directory = os.path.join(args.output, args.name)
if not os.path.exists(directory):
    os.makedirs(directory)
os.chdir(directory)

# Download and extract
url = "https://ida191.csc.fi:4430/download?token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJleHAiOjE2NTU3OTYwMjEsImRhdGFzZXQiOiIyMmI1MTkxZS1mMjRiLTQ0NTctOTNkMy05NTc5N2M5MDBmYzAiLCJwYWNrYWdlIjoiMjJiNTE5MWUtZjI0Yi00NDU3LTkzZDMtOTU3OTdjOTAwZmMwX3VpNjV6aXBrLnppcCJ9.00GP5nsjsH1KnnLVeVKMQhUxM53SAs1AKJreGiUnp3A&dataset=22b5191e-f24b-4457-93d3-95797c900fc0&package=22b5191e-f24b-4457-93d3-95797c900fc0_ui65zipk.zip"
archive = '22b5191e-f24b-4457-93d3-95797c900fc0_ui65zipk.zip'
utils.download_url(url, archive)
utils.extract_archive(archive, delete=True)
utils.extract_archive(os.path.join('SealID', 'full images.zip'), delete=True)
utils.extract_archive(os.path.join('SealID', 'patches.zip'), delete=True)