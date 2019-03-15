import numpy as np
import cv2
import os
import dlib
folderName = os.path.join(os.path.dirname(__file__), 'ncre')
ext = '.jpg'
fileList = os.listdir(folderName)
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
    mask[yMin-10:, xMin+5:xMax-5] = cv2.GC_FGD
    mask[0:30, 0:30] = cv2.GC_BGD
    mask[0:30, len(img[0])-30:len(img[0])] = cv2.GC_BGD
    return True

if not os.path.exists(folderName + '_seg'):
    os.mkdir(folderName + '_seg')
if not os.path.exists(folderName + '_mask'):
    os.mkdir(folderName + '_mask')
for name in fileList:
    print("proccessing " + name)
    if not name.endswith(ext):
        continue
    img = cv2.imread(os.path.join(folderName, name))
    # 这里假定原图大部分为前景
    mask = np.ones(img.shape[:2],np.uint8) * cv2.GC_PR_BGD
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)
    if not setMask(img, mask):
        print("error on " + name)
        continue
    cv2.grabCut(img,mask,None,bgdModel,fgdModel,10,cv2.GC_INIT_WITH_MASK)
    mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8') * 255
    cv2.imwrite(os.path.join(folderName + '_seg', name), mask2)
    maskImg = img
    maskImg[mask2==0] = (0,0,0)
    cv2.imwrite(os.path.join(folderName + '_mask', name), maskImg)