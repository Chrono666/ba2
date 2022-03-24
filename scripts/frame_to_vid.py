import cv2
import numpy as np
import glob

img_array = []
for filename in glob.glob('C:/Users/User/projects/school/ba2/data/test/*.jpg'):
    img = cv2.imread(filename)
    img = cv2.resize(img, (224, 224))
    height, width, layers = img.shape
    size = (width, height)
    img_array.append(img)

out = cv2.VideoWriter('project.avi', cv2.VideoWriter_fourcc(*'DIVX'), 15, size)

for i in range(len(img_array)):
    out.write(img_array[i])
out.release()