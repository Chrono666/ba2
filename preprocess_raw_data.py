import argparse
import os

from utils.converters import level_out_dataset_classes, reduce_data, split_raw_data_into_train_val_test
from utils.remove_duplicates import remove_duplicates

parser = argparse.ArgumentParser(description='raw data preprocessing')

parser.add_argument(
    '--data-dir',
    type=str,
    metavar='DD',
    help='path to data on which classification is to be performed')
parser.add_argument(
    '--reduce-frames',
    default=False,
    type=bool,
    metavar='RF',
    help='if frames in folder should be reduced to resemble n frames per second')
parser.add_argument(
    '--frame-rate',
    default=10,
    type=int,
    metavar='FR',
    help='frame rate of reduced frames if reduced frame argument is set')
parser.add_argument(
    '--remove-duplicates',
    default=False,
    type=bool,
    metavar='RF',
    help='if duplicates should be removed (default: False)')
parser.add_argument(
    '--downsample',
    default=False,
    type=bool,
    metavar='LO',
    help='deletes images in folder till both classes contain equal amount of images')
parser.add_argument(
    '--split-folders',
    default=False,
    type=bool,
    metavar='SF',
    help='if data should be split into train, val, test folders')
parser.add_argument(
    '--split-ratio',
    type=int,
    default=(0.8, 0.1, 0.1),
    metavar='SR',
    help='ratio of train, val, test data')

args = parser.parse_args()

if __name__ == '__main__':
    if args.reduce_frames:
        reduce_data(args.data_dir, args.frame_rate)

    if args.remove_duplicates:
        for root, dirs, files in os.walk(args.data_dir):
            for folder in dirs:
                remove_duplicates(os.path.join(root, folder))

    if args.downsample:
        level_out_dataset_classes(args.data_dir)

    if args.split_folders:
        split_raw_data_into_train_val_test(args.data_dir, args.split_ratio)

    print('Done')
