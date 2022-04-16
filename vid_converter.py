import argparse
import os

from tensorflow.python.keras.applications.vgg16 import preprocess_input

from model import dataset
from utils.converters import convert_video_to_frames

parser = argparse.ArgumentParser(description='raw data preprocessing')

parser.add_argument(
    '--data-dir',
    type=str,
    metavar='DD',
    help='path to video data')
parser.add_argument(
    '--img-dir',
    type=str,
    metavar='DD',
    help='path to image data')
args = parser.parse_args()

if __name__ == '__main__':
    convert_video_to_frames(args.data_dir, args.img_dir)
