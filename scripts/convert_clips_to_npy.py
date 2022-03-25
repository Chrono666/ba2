import numpy as np
import skvideo.io
import glob

# important clips have to be the same size (frames)
npy_array = []
for filename in glob.glob('C:/Users/User/projects/school/ba2/data/video_test/*.avi'):
    videodata = skvideo.io.vread(filename)
    npy_array.append(videodata)

print(np.array(npy_array).shape)

np.save('test', np.array(npy_array))
