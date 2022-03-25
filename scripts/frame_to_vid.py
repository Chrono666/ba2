import cv2
import glob

img_array = []
image_array = []
arr = []
counter = 1
for filename in glob.glob('C:/Users/User/projects/school/ba2/data/test/*.jpg'):
    img = cv2.imread(filename)
    img = cv2.resize(img, (224, 224))
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height, width, layer = img.shape
    size = (width, height)
    arr.append(img)
    if len(arr) % 20 == 0:
        # for generate greyscale video
        #out = cv2.VideoWriter((str(counter) + '.avi'), cv2.VideoWriter_fourcc(*'DIVX'), 20, (width, height), 0)
        # for generate rgb video
        out = cv2.VideoWriter(('./data/video_test/'+str(counter) + '.avi'), cv2.VideoWriter_fourcc(*'DIVX'), 60, size)
        counter = counter + 1
        for i in range(len(arr)):
            out.write(arr[i])
        arr = []
        out.release()

