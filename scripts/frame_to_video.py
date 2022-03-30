import cv2
import glob


def convert_image_to_video(input_path, output_path, new_frame_width, new_frame_height, video_format='avi', frames=60,
                           split=False):
    arr = []
    count = 1
    for filename in glob.glob('{}/*.jpg'.format(input_path)):
        img = cv2.imread(filename)
        img = cv2.resize(img, (new_frame_width, new_frame_height))
        height, width, layer = img.shape
        size = (width, height)
        arr.append(img)
        if split:
            if len(arr) % 20 == 0:
                out = video_converter(output_path, size, video_format, frames, count)
                count += 1
                for i in range(len(arr)):
                    out.write(arr[i])
                arr = []
                out.release()
        else:
            out = video_converter(output_path, size, video_format, frames, count)
            for i in range(len(arr)):
                out.write(arr[i])
            out.release()


def video_converter(output_path, size, video_format='avi', frames=60, counter=1):
    if video_format == 'avi':
        return cv2.VideoWriter(('{}/{}.avi'.format(output_path, counter)), cv2.VideoWriter_fourcc(*'DIVX'), frames,
                               size)
    elif video_format == 'mp4':
        return cv2.VideoWriter(('{}/{}.mp4'.format(output_path, counter)), cv2.VideoWriter_fourcc(*'mp4v'), frames,
                               size)
    else:
        print('Select valid video format, either avi or mp4')
        return None
