# -*- coding: utf-8 -*- 
from src.util import faceFromDir

IN_DIR = 'datasets/origin'
OUT_DIR = 'datasets/aligned'

SHAPE_MODEL = "models/shape_predictor_68_face_landmarks.dat"

faceFromDir(IN_DIR, OUT_DIR, SHAPE_MODEL)