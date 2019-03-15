import cv2
import os

folder = 'George_W_Bush'
folder_dest = 'origin_128'
if not os.path.isdir(folder_dest):
    os.mkdir(folder_dest)
for file in os.listdir(folder):
    img = cv2.imread(os.path.join(folder, file))
    resizeImg = cv2.resize(img, (128,128))
    cv2.imwrite(os.path.join(folder_dest, file), resizeImg)