import os
import cv2
import dlib
import numpy as np

folder = '../images/facenet/George_W_Bush'
destFolder = '../images/facenet/George_W_Bush_test'

shape_model = os.path.join(os.path.dirname(__file__), '../../models/shape_predictor_68_face_landmarks.dat')
shapePredict = dlib.shape_predictor(shape_model)
detector = dlib.get_frontal_face_detector()

def setMask(img, mask):
    dets = detector(img, 1)
    if len(dets) == 0:
        return False
    shape = shapePredict(img, dets[0])
    xMin = len(img[0])
    xMax = 0
    yMin = len(img)
    yMax = 0
    for i in range(shape.num_parts):
        if (shape.part(i).x < xMin):
            xMin = shape.part(i).x
        if (shape.part(i).x > xMax):
            xMax = shape.part(i).x
        if (shape.part(i).y < yMin):
            yMin = shape.part(i).y
        if (shape.part(i).y > yMax):
            yMax = shape.part(i).y
    mask[yMin:yMax, xMin:xMax] = (255, 255, 255)
    return True

if not os.path.isdir(destFolder):
    os.mkdir(destFolder)

for file in os.listdir(folder):
    print("processing " + file)
    path = os.path.join(folder, file)
    img = cv2.imread(path)
    mask_fg = np.zeros(img.shape, np.uint8)
    setMask(img, mask_fg)
    mask_bg = np.zeros(img.shape, np.uint8)
    mask_bg[0:30, len(img[0])-30:len(img[0])] = (255,255,255)
    mask_bg[0:30, 0:30] = (255,255,255)
    cv2.imwrite(os.path.join(destFolder, file[:-4] + '.png'), img)
    cv2.imwrite(os.path.join(destFolder, file[:-4] + '_bg.png'), mask_bg)
    cv2.imwrite(os.path.join(destFolder, file[:-4] + '_fg.png'), mask_fg)
