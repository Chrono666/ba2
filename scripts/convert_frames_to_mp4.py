import cv2
import glob

img_array = []
for filename in glob.glob('C:/Users/User/projects/school/ba2/data/data_raw/OK/*.jpg'):
    img = cv2.imread(filename)
    img = cv2.resize(img, (160, 128))
    height, width, layers = img.shape
    size = (width, height)
    img_array.append(img)

out = cv2.VideoWriter('def.mp4', cv2.VideoWriter_fourcc(*'MP4V'), 60, size)

for i in range(len(img_array)):
    out.write(img_array[i])
out.release()
