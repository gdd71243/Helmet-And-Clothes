import datetime
import os
import threading

import cv2
import torch

from utils.plots import plot_one_box
from . import alert_util
from PIL import Image
import numpy as np
import random

from utils.torch_utils import select_device, load_classifier, time_synchronized
from utils.general import (scale_coords, clip_coords, xywh2xyxy, xyxy2xywh)
import time

from loguru import logger

from models.transreid.predict import Predict

def save_image(dir, image_count, label, image): #保存图片
    file_name = str(label).zfill(4) + '_c' + str(1) + '_f' + str(image_count).zfill(7) + '.jpg'
    file_name = os.path.join(dir, file_name)
    image.save(file_name)

def get_image_count(file_path): #得到已存图片最大 接下来图片顺延
    file_count = 1300000
    files = os.listdir(file_path)
    for fi in files:
        file_count = max(file_count, int(fi[-11:-4]))
    return file_count + 1

def save_one_box(xyxy, im, gain=1.02, pad=10, square=False, BGR=False): #裁剪出人
    xyxy = torch.tensor(xyxy).view(-1, 4)
    crop = im[int(xyxy[0, 1]):int(xyxy[0, 3]), int(xyxy[0, 0]):int(xyxy[0, 2])]
    return crop

class frame_buffer:
    # 缓存若干帧
    def __init__(self, fps=5, buffer_s=5, base_name='', save_dir='./video/', cam_id=1, detect_list=None, names=None,
                 colors=None, img_rescaled_size=None, strategy_util=None, predict=None, safe_clothes_list=None):
        if names is None:
            names = []
        if detect_list is None:
            detect_list = [1]
        if colors is None:
            colors = [[250, 125, 20], [55, 60, 252], [105, 217, 82]]

        self.fps = fps
        self.buffer_time = buffer_s  # buffer的时间长度
        self.buffer_size = self.fps * self.buffer_time
        
        self.frames = []  # frames[i]=(frame, det, confidence)
        self.unprocessed_frames = []  # unprocessed_frames[i]=(frame, det, None)
        # self.confidences = []
        self.base_name = base_name
        self.dir = save_dir
        # self.threshold = threshold
        self.snapshot = None  # snapshot 会定格在最后一个有目标事件的帧
        self.cam_id = cam_id

        self.wait_for_check = self.fps * 5
        self.cold_count = self.fps * 9999

        self.detect_list = detect_list
        self.names = names
        self.colors = colors
        self.img_rescaled_size = img_rescaled_size
        self.strategy_util = strategy_util

        self.predict = predict
        self.evaluator = self.predict.get_evaluator()
        self.count = 0
        self.safe_clothes_list = safe_clothes_list
        self.dir = './runs/crop_image'
        threading.Thread(target=self.process_frame).start()

    def process_frame(self):
        while True:
            if self.unprocessed_frames:
                
                frame, det, _ = self.unprocessed_frames.pop(0)
                # if int(self.cam_id )== 646:
                #     print(frame.shape, type(frame))
                #     print("saving")
                #     img_Image = frame.astype(np.uint8)
                #     img_Image = Image.fromarray(cv2.cvtColor(img_Image, cv2.COLOR_BGR2RGB))
                #     save_dir = os.path.join('./runs/temp', str(self.count).zfill(4) + '.jpg')
                #     img_Image.save(save_dir)
                #     self.count += 1
                image_list = []
                index_list = []
                max_conf = 0
                have_person = False
                self.count += 1
                if len(det):
                    det[:, :4] = scale_coords(self.img_rescaled_size, det[:, :4], frame.shape).round()
                    # calculate results
                    l = len(det) - 1
                    for i, (*xyxy, conf, cls) in enumerate(reversed(det)):
                        if int(cls) == 0:
                            have_person = True
                            croped_img = save_one_box(xyxy, frame, BGR=True)
                            img_Image = Image.fromarray(cv2.cvtColor(croped_img, cv2.COLOR_BGR2RGB))
                            image_list.append(img_Image)
                            index_list.append(l-i)
                            # det[l-i, 5] = int(self.predict.predict_from_image(image_list, self.evaluator)[0]) + 3 # 3 是 class num
                            
                        max_conf = max(max_conf, (int(cls) in self.detect_list) * float(conf))
                        
                    if image_list != []:
                        labels = self.predict.predict_from_image(image_list, self.evaluator)
                        for i, index in enumerate(index_list):
                            #save_image(self.dir , get_image_count(self.dir), labels[i], image_list[i])
                            label_temp = int(labels[i])
                            if label_temp in self.safe_clothes_list:
                                label_temp = 3
                            else:
                                label_temp = 4
                            det[index, 5] = label_temp
                            max_conf = max(max_conf, (label_temp-3) * float(conf))


                confidence = have_person * max_conf

                frame_struc = (frame, det, confidence)

                self.frames.append(frame_struc)
                self.frames = self.frames[-self.buffer_size:]

                if confidence > 0:
                    self.snapshot = frame_struc

                if str(self.cam_id) == str(self.strategy_util.config['show_cam']) and self.frames:
                    alert_util.IMG_POSTING_BUFFER.append(self.draw_box(self.frames[-1]))

                self.check()
            time.sleep(0.04)

    def draw_box(self, frame_struc):
        frame, det, confidence = frame_struc
        # 当确定图中的框需要绘制时
        if confidence > 0 and len(det):
            # Rescale boxes from img_size to im0 size
            # det[:, :4] = scale_coords(
            #     self.img_rescaled_size, det[:, :4], frame.shape).round()

            for *xyxy, conf, cls in reversed(det):
                if int(cls) != 0:
                    label = f'{self.names[int(cls)]} {conf:.2f}'
                    # plot_one_box(xyxy, frame, label=label,
                    #              color=self.colors[int(cls)], line_thickness=3)
                    plot_one_box(xyxy, frame, label=label,
                                color=self.colors(int(cls), True), line_thickness=3)
        return frame

    def insert_img(self, frame):
        """
        frame: (frame, det, None)
        """
        self.unprocessed_frames.append(frame)
        # self.confidences.append(conf)

    def check(self):
        # 当没有目标或冷却时间已超时，wait_for_check
        if self.frames[-1][-1] < 0.1 or self.cold_count > self.fps * 120:
            self.wait_for_check -= 1
        # wait_for_check小于0时可以开始检测
        if self.wait_for_check < 0:
            avg_confidences = sum([_[-1] for _ in self.frames]) / len(self.frames)
            if len(self.frames) >= self.fps * self.buffer_time and \
                    avg_confidences > self.strategy_util.config['threshold']:
                threading.Thread(target=self.raise_alert).start()
                self.wait_for_check = self.fps * 5
                self.cold_count = 0
        # wait_for_check大于0时没有在检测，计入冷却时间
        else:
            self.cold_count += 1

    @logger.catch()
    def raise_alert(self):
            # 取得当前的缓存帧
            frames = self.frames
            self.frames = []

            # 发送警报
            snapshot = self.draw_box(self.snapshot)
            success, encode_snp = cv2.imencode(".jpg", snapshot)

            encode_bytes = encode_snp.tobytes()
            alert_id = self.strategy_util.post_alert(encode_bytes, video=None, cam_id=self.cam_id, level=1)

            # 单独处理视频
            logger.info('saving video...')
            snp_path, vid_path = self.get_path()
            vid_writer = None

            frames = [self.draw_box(_) for _ in frames]
            if frames:
                fps = self.fps
                sp = frames[0].shape
                w = sp[1]
                h = sp[0]
                vid_writer = cv2.VideoWriter(
                    vid_path, cv2.VideoWriter_fourcc(*'VP90'), fps, (w, h))

            for img in frames:
                vid_writer.write(img)
            vid_writer.release()

            with open(vid_path, 'rb') as f:
                video = f.read()
            self.strategy_util.post_alert_video(alert_id, video)
            os.remove(vid_path)

    def get_f_path(self):
        time_sign = datetime.datetime.now().strftime('%y%m%d%H%M%S%A')
        return self.dir + 'tmp_frame_' + time_sign + '_' + str(time_synchronized()).split('.')[-1] + '.jpg'

    def get_path(self):
        time_sign = datetime.datetime.now().strftime('%y%m%d%H%M%S%A')
        return self.dir + self.base_name + 'video_' + time_sign + '.jpg', self.dir + self.base_name + 'snap_' + time_sign + '.webm'
