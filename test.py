import argparse
import datetime
import os
import shutil
import threading
from threading import Semaphore, Lock, Thread

import time
from pathlib import Path
from turtle import color
from utils import alert_util

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
# from utils.datasets import LoadStreams, LoadImages, LoadStreamsLive
from utils.load_stream import LoadStreamsLive
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized
from utils.buffer import frame_buffer
from loguru import logger

with torch.no_grad():
    device = select_device('0')
    half = True # half precision only supported on CUDA

    weights = ['weights/helmet_head_person_m.pt']
    
    imgsz=640
    models = [attempt_load(_, map_location=device) for _ in weights]  # load FP32 model
    strides = [int(model.stride.max()) for model in models]  # model stride
    imgszs = [check_img_size(imgsz, s=stride) for stride in strides]  # check img_size
    cudnn.benchmark = True
    for model in models:
        if half:
            model.half()  # to FP16


    for model in models:

        # Get names and colors
        names = model.module.names if hasattr(model, 'module') else model.names


        # Run inference
        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(
                next(model.parameters())))  # run once

    while True:
        time.sleep(1)