# -*- coding: utf-8 -*- 
from src.util import headFromDir

IN_DIR = 'datasets/George_W_Bush'
OUT_DIR = 'datasets/George_W_Bush_test'

SHAPE_MODEL = "models/shape_predictor_68_face_landmarks.dat"
IMG_SIZE = 128
FACE_SIZE = 64

headFromDir(IN_DIR, OUT_DIR, SHAPE_MODEL, IMG_SIZE, FACE_SIZE, 10, 30)