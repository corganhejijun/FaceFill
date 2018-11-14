# -*- coding: utf-8 -*- 
from src.util import faceFromDir

IN_DIR = 'datasets/origin'
OUT_DIR = 'datasets/aligned'

SHAPE_MODEL = "models/shape_predictor_68_face_landmarks.dat"
IMG_SIZE = 256
FACE_SIZE = 128

faceFromDir(IN_DIR, OUT_DIR, SHAPE_MODEL, IMG_SIZE, FACE_SIZE)