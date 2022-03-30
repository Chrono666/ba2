from scripts.frame_to_video import convert_image_to_video
from scripts.video_to_frame import convert_video_to_frames

# for filename in glob.glob('C:/Users/User/projects/school/ba2/data/test/*.jpg'):

if __name__ == '__main__':
    # print('Convert Video to frames')
    # convert_video_to_frames('./data/data_video/Aufnahmen Wärmebild 23.03.2022 neue Farbskala/MyRecord2022-03-23T122437136.avi', './data/test/frames_from_video')
    print('Convert frames to video')
    convert_image_to_video('./data/test/frames_from_video', './data/test/video_from_frames', 160, 128, 'mp4', frames=10)
    print('Done')
