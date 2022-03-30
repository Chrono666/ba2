import os
import cv2


def convert_video_to_frames(file_path, output_path, new_frame_width=None, new_frame_height=None):
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
