import numpy as np
import skvideo.io
import glob


# important clips have to be the same size (frames)
def convert_video_to_npy(input_path, video_format):
    npy_array = []
    for filename in glob.glob('{}/*.{}'.format(input_path, video_format)):
        video_data = skvideo.io.vread(filename)
        npy_array.append(video_data)

    print(np.array(npy_array).shape)
    np.save('test', np.array(npy_array))
