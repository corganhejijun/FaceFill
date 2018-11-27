# -*- coding: utf-8 -*- 
import cv2
import os
import dlib
from scipy import misc
import numpy as np
from PIL import Image

def getBound(img, shape):
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
    return xMin, xMax, yMin, yMax
    
def combineImg(imga, imgb):
    target = Image.new('RGB', (imga.shape[0]*2, imga.shape[1]))
    target.paste(Image.fromarray(np.uint8(imga)), (0, 0))
    target.paste(Image.fromarray(np.uint8(imgb)), (imgb.shape[0] + 1, 0))
    return target

def getFace(detector, shapePredict, img):
    dets = detector(img, 1)
    if (len(dets) == 0):
        return None, None, None, None
    det = dets[0]
    shape = shapePredict(img, det)
    xmin, xmax, ymin, ymax = getBound(img, shape)
    if xmin < 0 or xmax < 0 or ymin < 0 or ymax < 0:
        return None, None, None, None
    return xmin, xmax, ymin, ymax

def headFromDir(inDir, outDir, shape_model, size, faceSize, outBleed_x=0, outBleed_y=0):
    shapePredict = dlib.shape_predictor(shape_model)
    detector = dlib.get_frontal_face_detector()
    if not os.path.exists(outDir):
        os.mkdir(outDir)
    count = 0
    fileList = os.listdir(inDir)
    for name in fileList:
        count += 1
        print("processing %s, current %d of total %d" % (name, count, len(fileList)))
        fileName = os.path.join(inDir, name)
        if not fileName.endswith('.jpg'):
            continue
        
        img = cv2.cvtColor(cv2.imread(fileName), cv2.COLOR_BGR2RGB)
        dets = detector(img, 1)
        if (len(dets) == 0):
            print("file %s has no face" % name)
            continue
        det = dets[0]
        shape = shapePredict(img, det)
        xmin, xmax, ymin, ymax = getBound(img, shape)
        if xmin < 0 or xmax < 0 or ymin < 0 or ymax < 0:
            print("file %s can't get bound" % name)
            continue

        left = xmin
        right = xmax
        top = ymin
        bottom = ymax
        longEdge = xmax - xmin
        shortEdge = ymax - ymin
        if longEdge < (ymax - ymin):
            longEdge = ymax - ymin
            shortEdge = xmax - xmin
            # To get square crop area, begin from face middle, take 1 facesize to the upward
            # take 0.5 facesize to the downward, take 1.5/2 facesize to the left and right respectively.
            top = int(ymin - longEdge)
            bottom = int(ymax + longEdge / 2)
            left = int(xmin - longEdge * 1.5 / 2)
            right = int(xmax + longEdge * 1.5 / 2)
        else:
            left = int(xmin - shortEdge * 1.5 / 2)
            right = int(xmax + shortEdge * 1.5 / 2)
            top = int(ymin - shortEdge)
            bottom = int(ymax + shortEdge / 2)

        if outBleed_x > 0:
            left -= outBleed_x
            right -= outBleed_x
        if outBleed_y > 0:
            top -= outBleed_y

        fullImg = np.zeros((size, size, 3))
        marginLeft = 0
        if left < 0:
            marginLeft = -int(left * size / (right - left))
            left = 0
        marginTop = 0
        if top < 0:
            marginTop = -int(top * size / (bottom - top))
            top = 0
        marginRight = 0
        if right > img.shape[1]:
            marginRight = int((right - img.shape[1]) * size / (right - left))
            right = img.shape[1]
        marginBottom = 0
        if bottom > img.shape[0]:
            marginBottom = int((bottom - img.shape[0]) * size / (bottom - top))
            bottom = img.shape[0]
        
        cropedImg = img[top:bottom, left:right, :]
        cropedImg = cv2.resize(cropedImg, dsize=(size - marginLeft - marginRight, size - marginTop - marginBottom))
        fullImg[marginTop : size - marginBottom, marginLeft : size - marginRight, :] = cropedImg
        if marginLeft > 0:
            fullImg[marginTop:(size - marginBottom), 0:marginLeft, :] = np.tile(np.reshape(cropedImg[:,0,:], (size - marginTop - marginBottom, 1, 3)), (1, marginLeft, 1))
        if marginRight > 0:
            fullImg[marginTop:(size - marginBottom), (size - marginRight):size, :] = np.tile(np.reshape(cropedImg[:, cropedImg.shape[1] - 1, :], (size - marginTop - marginBottom, 1, 3)), (1, marginRight, 1))
        if marginTop > 0:
            fullImg[0:marginTop, :, :] = np.tile(np.reshape(fullImg[marginTop, :, :], (1, size, 3)), (marginTop, 1, 1))
        if marginBottom > 0:
            fullImg[(size - marginBottom):size, :, :] = np.tile(np.reshape(fullImg[(size - marginBottom), :, :], (1, size, 3)), (marginBottom, 1, 1))

        fullFace = np.zeros((size, size, 3))
        xminFace, xmaxFace, yminFace, ymaxFace = getFace(detector, shapePredict, fullImg.astype(np.uint8))
        fullFace[yminFace:ymaxFace, xminFace:xmaxFace, :] = fullImg[yminFace:ymaxFace, xminFace:xmaxFace, :]
        if xminFace == None:
            print("file %s can't get face in fullImg" % name)
            continue
        combine = combineImg(fullImg, fullFace)
        outPath = os.path.join(outDir, str(count).zfill(6) + '.jpg')
        misc.imsave(outPath, combine)
