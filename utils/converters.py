import glob
import os
import random

import cv2
import numpy as np
import skvideo.io
import splitfolders
from tqdm import tqdm


def convert_image_to_video(input_path, output_path, new_frame_width, new_frame_height, video_format='avi', frames=60,
                           split=False):
    """Convert frames to video file

    Arguments:
        input_path {str} -- path to frames
        output_path {str} -- path to output video
        new_frame_width {int} -- frame width
        new_frame_height {int} -- frame height
        video_format {str} -- video format currently only avi and mp4 are supported
        frames {int} -- frames per second that are used in the video
        split {bool} -- if True, the video will be split into multiple videos
    """
    arr = []
    for filename in glob.glob('{}/*.jpg'.format(input_path)):
        img = cv2.imread(filename)
        print(filename)
        img = cv2.resize(img, (new_frame_width, new_frame_height))
        height, width, layer = img.shape
        size = (width, height)
        arr.append(img)
        if split:
            count = 1
            if len(arr) % 20 == 0:
                out = video_converter(output_path, size, video_format, frames, count)
                count += 1
                for i in range(len(arr)):
                    out.write(arr[i])
                arr = []
                out.release()
        else:
            out = video_converter(output_path, size, video_format, frames)
            for i in range(len(arr)):
                out.write(arr[i])
            out.release()


def video_converter(output_path, size, video_format='avi', frames=60, counter=1):
    """Convert frames to video file

    Arguments:
        output_path {str} -- path to output video
        size {tuple} -- witdt and height of the frames
        video_format {str} -- video format currently only avi and mp4 are supported
        frames {int} -- frames per second that are used in the video

    Returns:
        [VideoWriter] -- VideoWriter object
    """
    if video_format == 'avi':
        return cv2.VideoWriter(('{}/{}.avi'.format(output_path, counter)), cv2.VideoWriter_fourcc(*'DIVX'), frames,
                               size)
    elif video_format == 'mp4':
        return cv2.VideoWriter(('{}/{}.mp4'.format(output_path, counter)), cv2.VideoWriter_fourcc(*'mp4v'), frames,
                               size)
    else:
        print('Select valid video format, either avi or mp4')
        return None


def convert_video_to_frames(file_path, output_path, new_frame_width=None, new_frame_height=None):
    """Convert video to frames

    Arguments:
        file_path {str} -- path to video
        output_path {str} -- path to output frames
        new_frame_width {int} -- frame width
        new_frame_height {int} -- frame height
    """
    file_name = os.path.basename(file_path)
    video_capture = cv2.VideoCapture(file_path)
    success, image = video_capture.read()
    count = 0
    while success:
        if new_frame_width is not None or new_frame_height is not None:
            image = cv2.resize(image, (new_frame_width, new_frame_height))
        cv2.imwrite("{}/{}{}.jpg".format(output_path, file_name, count), image)  # save frame as JPEG file
        success, image = video_capture.read()
        count += 1


# important clips have to be the same size (frames)
def convert_video_to_npy(input_path, video_format):
    """Convert video to numpy array

    Arguments:
        input_path {str} -- path to video
        video_format {str} -- video format to build path to video file
    """
    npy_array = []
    for filename in glob.glob('{}/*.{}'.format(input_path, video_format)):
        video_data = skvideo.io.vread(filename)
        npy_array.append(video_data)

    print(np.array(npy_array).shape)
    np.save('test', np.array(npy_array))


def convert_avi_to_mp4(avi_file_path, output_name):
    os.popen(
        "ffmpeg -i {input} -ac 2 -b:v 2000k -c:a aac -c:v libx264 -b:a 160k -vprofile high -bf 0 -strict experimental "
        "-f mp4 {output}.mp4".format(
            input=avi_file_path, output=output_name))
    return True


def reduce_data(input_path, frame_rate):
    """Load images from folder and deletes all images that do not fulfill i % 60 == 0 and saves them to new folder

    Arguments:
        input_path {str} -- path to images
        frame_rate {int} -- reduce images to this frame rate
    """
    for root, dirs, files in os.walk(input_path):
        for folder in dirs:
            for i, filename in enumerate(glob.glob('{}/*.jpg'.format(os.path.join(input_path, folder)))):
                if i % frame_rate != 0:
                    os.remove(filename)
    print('All frames that are not 0 when % {} have been deleted'.format(frame_rate))


def level_out_dataset_classes(input_path):
    """ Compare number of images of two folders and deletes the difference from the bigger folder.

    Arguments:
        input_path {str} -- directory path
    """
    ok_image_size = 0
    def_image_size = 0
    ok_image_files = []
    def_image_files = []
    ok_image_path = ''
    def_image_path = ''
    for root, dirs, files in os.walk(input_path):
        for folder in dirs:
            if folder == 'DEF':
                def_image_path = os.path.join(root, folder)
                def_image_size = len(os.listdir(def_image_path))
                def_image_files = os.listdir(def_image_path)
            if folder == 'OK':
                ok_image_path = os.path.join(root, folder)
                ok_image_size = len(os.listdir(ok_image_path))
                ok_image_files = os.listdir(ok_image_path)
        if ok_image_size > def_image_size:
            print('OK folder has more images than DEF folder')
            files = random.sample(ok_image_files, abs(ok_image_size - def_image_size))
            for file in tqdm(files):
                os.remove(os.path.join(ok_image_path, file))
            return
        if ok_image_size < def_image_size:
            print('DEF folder has more images than OK folder')
            files = random.sample(def_image_files, abs(ok_image_size - def_image_size))
            for file in tqdm(files):
                os.remove(os.path.join(def_image_path, file))
            return


def split_raw_data_into_train_val_test(input_path, ratio=(0.7, 0.15, 0.15)):
    """Split raw data into train, validation and test set.

    Arguments:
        input_path {str} -- path to raw data
        ratio {tuple} -- ratio of train, validation and test set
    """
    splitfolders.ratio(input_path, output=os.path.join(input_path, 'balanced_data'),
                       seed=1337, ratio=ratio, group_prefix=None, move=False)  # default values
    print('Data has been split into train, val and test')
