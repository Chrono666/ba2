import skvideo.io
videodata = skvideo.io.vread("project.avi")
print(videodata.shape)
