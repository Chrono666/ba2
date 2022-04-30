import argparse
from utils.converters import convert_video_to_frames, generate_image_patches

parser = argparse.ArgumentParser(description='raw data preprocessing')

parser.add_argument(
    '--input-dir',
    type=str,
    metavar='DD',
    required=True,
    help='input path')
parser.add_argument(
    '--out-dir',
    type=str,
    metavar='DD',
    required=True,
    help='output path')
args = parser.parse_args()

if __name__ == '__main__':
    # convert_video_to_frames(args.data_dir, args.img_dir)

    """patch coordinates for 06.11.2022"""
    # generate_image_patches(args.input_dir, args.out_dir, coordinates=[15, 105, 0, 85], patch_name='circle_first')
    # generate_image_patches(args.input_dir, args.out_dir, coordinates=[22, 107, 80, 168], patch_name='cross_first')
    # generate_image_patches(args.input_dir, args.out_dir, coordinates=[30, 107, 165, -1], patch_name='triangle_first')
    # generate_image_patches(args.input_dir, args.out_dir, coordinates=[113, 200, 0, 80], patch_name='circle_middle')
    # generate_image_patches(args.input_dir, args.out_dir, coordinates=[115, 205, 80, 168], patch_name='cross_middle')
    # generate_image_patches(args.input_dir, args.out_dir, coordinates=[115, 205, 164, -1], patch_name='triangle_middle')
    # generate_image_patches(args.input_dir, args.out_dir, coordinates=[210, 302, 0, 80], patch_name='circle_last')
    # generate_image_patches(args.input_dir, args.out_dir, coordinates=[212, 305, 80, 167], patch_name='cross_last')
    # generate_image_patches(args.input_dir, args.out_dir, coordinates=[220, 304, 163, -1], patch_name='triangle_last')

    """patch coordinates for 11.03.2022"""
    # generate_image_patches(args.input_dir, args.out_dir, coordinates=[0, 77, 2, 87], patch_name='circle_first')
    # generate_image_patches(args.input_dir, args.out_dir, coordinates=[0, 75, 85, 175], patch_name='cross_first')
    # generate_image_patches(args.input_dir, args.out_dir, coordinates=[0, 75, 180, -1], patch_name='triangle_first')
    # generate_image_patches(args.input_dir, args.out_dir, coordinates=[108, 192, 0, 82], patch_name='circle_middle')
    # generate_image_patches(args.input_dir, args.out_dir, coordinates=[100, 188, 85, 175], patch_name='cross_middle')
    # generate_image_patches(args.input_dir, args.out_dir, coordinates=[110, 185, 180, -1], patch_name='triangle_middle')
    # generate_image_patches(args.input_dir, args.out_dir, coordinates=[210, 300, 0, 82], patch_name='circle_last')
    # generate_image_patches(args.input_dir, args.out_dir, coordinates=[212, 300, 85, 175], patch_name='cross_last')
    # generate_image_patches(args.input_dir, args.out_dir, coordinates=[215, 300, 180, -1], patch_name='triangle_last')

    """patch coordinates for 23.03.2022"""
    generate_image_patches(args.input_dir, args.out_dir, coordinates=[15, 100, 0, 95], patch_name='circle_first')
    generate_image_patches(args.input_dir, args.out_dir, coordinates=[10, 105, 95, 178], patch_name='cross_first')
    generate_image_patches(args.input_dir, args.out_dir, coordinates=[20, 105, 178, -1], patch_name='triangle_first')
    generate_image_patches(args.input_dir, args.out_dir, coordinates=[105, 200, 0, 95], patch_name='circle_middle')
    generate_image_patches(args.input_dir, args.out_dir, coordinates=[105, 200, 95, 181], patch_name='cross_middle')
    generate_image_patches(args.input_dir, args.out_dir, coordinates=[110, 200, 179, -1], patch_name='triangle_middle')
    generate_image_patches(args.input_dir, args.out_dir, coordinates=[210, 300, 0, 95], patch_name='circle_last')
    generate_image_patches(args.input_dir, args.out_dir, coordinates=[210, 300, 95, 184], patch_name='cross_last')
    generate_image_patches(args.input_dir, args.out_dir, coordinates=[220, 300, 184, -1], patch_name='triangle_last')
