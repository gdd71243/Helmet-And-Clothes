# Dataset utils and dataloaders

import glob
import math
import os
import random
import shutil
import time
from itertools import repeat
from multiprocessing.pool import ThreadPool
from pathlib import Path
from threading import Thread, Semaphore
from multiprocessing import Process
from functools import wraps
from multiprocessing import Pool

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ExifTags
from torch.utils.data import Dataset
from tqdm import tqdm

from utils.general import xyxy2xywh, xywh2xyxy, xywhn2xyxy, xyn2xy, segment2box, segments2boxes, resample_segments, \
    clean_str
from utils.torch_utils import torch_distributed_zero_first

from loguru import logger
from utils.datasets import letterbox
from utils.torch_utils import select_device


class LoadStreamsLive:  # multiple IP or RTSP cameras
    def __init__(self, sources=None, img_size=640, stride=32):
        self.continue_sign = True

        if sources is None:
            sources = []
        self.img_size = img_size
        self.stride = stride

        n = len(sources)
        self.imgs = [None] * n
        # clean source names for later
        self.sources = [clean_str(x) for x in sources]
        self.caps = [None] * len(sources)
        self.update_threads = [True] * len(sources)

        def start_fetch(i, s, n):
            # Start the thread to read frames from the video stream
            logger.info(f'{i + 1}/{n}: {s}... ')
            cap = cv2.VideoCapture(eval(s) if s.isnumeric() else s, cv2.CAP_FFMPEG)
            assert cap.isOpened(), f'Failed to open {s}'
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS) % 100
            _, __ = cap.read()  # guarantee first frame
            if _:
                self.imgs[i] = __
                thread = Thread(target=self.update,
                                args=([i, cap]), daemon=True)
                self.caps[i] = cap
                self.update_threads[i] = True
                logger.info(f' success {i + 1}: ({w}x{h} at {fps:.2f} FPS).')
                thread.start()
            else:
                logger.info(f'fail to get: {i + 1}: {s}')

        for i, s in enumerate(sources):
            self.imgs[i] = np.ones((640, 640, 3)) * 200
            Thread(target=start_fetch, args=(i, s, n)).start()

        self.update_next = True
        self.rect = False

        self.sem = Semaphore(1)

        Thread(target=self.process_next_frame).start()
        time.sleep(1)
        self.rescaled_size = self.img.shape[2:]
        print('')  # newline

        # check for common shapes
        # s = np.stack([letterbox(x, self.img_size, stride=self.stride)[0].shape for x in self.imgs], 0)  # shapes
        # self.rect = np.unique(s, axis=0).shape[0] == 1  # rect inference if all shapes equal
        # if not self.rect:
        #     print('WARNING: Different stream shapes detected. For optimal performance supply similarly-shaped streams.')

    def reload_streams(self, sources=None):
        self.continue_sign = False
        while sum(self.update_threads) > 0:
            time.sleep(0.05)
        for c in self.caps:
            c.release()
        time.sleep(1)
        self.caps = []
        self.update_threads = []
        if sources is None:
            sources = []

        self.__init__(sources, self.img_size, self.stride)

    def update(self, index, cap):
        # Read next stream frame in a daemon thread
        n = 0
        while self.continue_sign and cap.isOpened():
            n += 1
            # _, self.imgs[index] = cap.read()
            cap.grab()
            if n == 20:  # read every 4th frame ==>改成了5
                success, im = cap.retrieve()
                self.imgs[index] = im if success else self.imgs[index] * 0
                n = 0
            time.sleep(0.01)  # wait time
        self.update_threads[index] = False

    def __iter__(self):
        self.count = -1
        return self

    @logger.catch()
    def process_next_frame(self):
        device = select_device('0')
        while True:
            self.sem.acquire()
            self.update_next = False

            # logger.debug("Processing next frame")

            img0 = self.imgs.copy()

            # Letterbox
            img = [letterbox(x, self.img_size, auto=self.rect, stride=self.stride)[
                       0] for x in img0]

            # Stack
            img = np.stack(img, 0)

            # Convert
            # BGR to RGB, to bsx3x416x416
            img = img[:, :, :, ::-1].transpose(0, 3, 1, 2)
            img = np.ascontiguousarray(img)
            img = torch.from_numpy(img).to(device)
            img = img.half()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            self.img = img
            self.img0 = img0

            self.update_next = False
            # logger.debug("Processed next frame")

    def __next__(self):
        self.count += 1

        if cv2.waitKey(1) == ord('q'):  # q to quit
            cv2.destroyAllWindows()
            raise StopIteration

        img = self.img
        img0 = self.img0
        self.sem.release()
        return self.sources, img, img0, self.caps

    def __len__(self):
        return 0  # 1E12 frames = 32 streams at 30 FPS for 30 years
