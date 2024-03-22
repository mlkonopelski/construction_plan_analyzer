# -*- coding: utf-8 -*-
import configparser
from roboflow import Roboflow
import cv2

VERSION = 1

FORMAT = 'coco-mmdetection' # yolov8, coco-mmdetection

config = configparser.ConfigParser()
config.sections()
config.read('config.ini')

rf = Roboflow(config['roboflow']['key'])
project = rf.workspace(config['roboflow']['workspace']).project(config['roboflow']['project'])

version = project.version(VERSION)
dataset = version.download(FORMAT)
