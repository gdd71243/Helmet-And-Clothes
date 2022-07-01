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
import PIL
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import numpy as np

from models.experimental import attempt_load
# from utils.datasets import LoadStreams, LoadImages, LoadStreamsLive
from utils.load_stream import LoadStreamsLive
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box, colors
from utils.torch_utils import select_device, load_classifier, time_synchronized
from utils.buffer import frame_buffer
from loguru import logger

from models.transreid.predict import Predict


DETECT_LIST = [1]  # 1是没戴安全帽，4是没穿安全服
SAFETY_CLOTHES = [1, 2, 3, 4, 5, 6, 8, 9, 10, 15, 17, 16, 23, 24, 25, 27, 29, 30, 31, 42, 45, 53, 54, 99, 101]

FPS = 1
BUFFER_SECOND = 5

logger.add('run.log', level='INFO')

time_sem = Semaphore(1)

def save_image(dir, image_count, label, image): #保存图片
    file_name = str(label).zfill(4) + '_c' + str(random.randint(1,5)) + '_f' + str(image_count).zfill(7) + '.jpg'
    file_name = os.path.join(dir, file_name)
    image.save(file_name)

def get_image_count(file_path): #得到已存图片最大 接下来图片顺延
    file_count = 1100000
    files = os.listdir(file_path)
    for fi in files:
        file_count = max(file_count, int(fi[-11:-4]))
    return file_count

@logger.catch()
def detect_timer():
    while True:
        try:
            time_sem.release()
            time.sleep(1 / FPS)
        except Exception:
            time.sleep(0.01)


@logger.catch()
def detect():
    source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size

    image_dir = './runs/crop_image'
    img_count = get_image_count(image_dir)
    save_flag = 0

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    weights = ['weights/helmet_head_person_m.pt']
    # weights = ['weights/helmet_head_person_m.pt','weights/clothes.pt']
    # weights = ['weights/clothes.pt']

    # Load model
    models = [attempt_load(_, map_location=device) for _ in weights]  # load FP32 model
    strides = [int(model.stride.max()) for model in models]  # model stride
    imgszs = [check_img_size(imgsz, s=stride) for stride in strides]  # check img_size

    for model in models:
        if half:
            model.half()  # to FP16

    predict = Predict(file_name='pickle.pkl', re_compute=False)

    strategy_util = alert_util.Strategy_util()

    url_list = []
    cam_list = []
    while not url_list:
        cam_list, url_list = strategy_util.parse_cam_info()
        time.sleep(0.1)

    # url_list=url_list[:30]
    # cam_list=cam_list+cam_list
    # Set Dataloader
    check_imshow()
    cudnn.benchmark = True  # set True to speed up constant image size inference
    dataset = LoadStreamsLive(url_list, img_size=imgsz, stride=strides[0])

    names_list = []
    class_count = []
    #colors = [[250, 125, 20], [55, 60, 252], [105, 217, 82], [10, 147, 150], [155, 34, 38]]
    
    for model in models:

        # Get names and colors
        names = model.module.names if hasattr(model, 'module') else model.names

        names_list.extend(names)
        class_count.append(len(names))

        # Run inference
        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(
                next(model.parameters())))  # run once
    
    
    names_list.append('safety clothes')     
    names_list.append('unsafety clothes')

    
    buffered_frames_list = [
        frame_buffer(fps=FPS, buffer_s=5, base_name='video', save_dir='./video/', cam_id=cam_list[_],
                     detect_list=DETECT_LIST, names=names_list, colors=colors, img_rescaled_size=dataset.rescaled_size,
                     strategy_util=strategy_util, predict=predict, safe_clothes_list=SAFETY_CLOTHES)
        for _
        in range(len(dataset.sources))]

    last_time = time_synchronized()

    # 根据 fps释放 sem 使下面的循环运行
    Thread(target=detect_timer).start()

    frame_count=0
    process_start = time_synchronized()

    for path, img, im0s, vid_cap in dataset:
        # while True: 
            frame_count+=1

            # 收到信号停止
            if not strategy_util.config['continue_run']:
                break

            logger.debug('finish preprocessing :' +
                        str((time_synchronized() - last_time)))

            # 用 sem 维持 fps 恒定
            time_sem.acquire()

            start_time = time_synchronized()

            pred_list = []

            # Inference
            for _, model in enumerate(models):
                t1 = time_synchronized()

                pred = model(img, augment=opt.augment)[0]

                # Apply NMS
                pred = non_max_suppression(
                    pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)

                # 将后续模型的cls 加上前面模型cls的总数
                for i in range(len(pred)):
                    if pred[i].numel():
                        pred[i][:, -1] += sum(class_count[:_])

                if not pred_list:
                    pred_list = pred
                else:
                    for i, (pl, cp) in enumerate(zip(pred_list, pred)):
                        pred_list[i] = torch.vstack([pl, cp])
                t2 = time_synchronized()

                logger.debug(f'pred cost {_}: ' + str(t2 - t1))

            logger.debug('finish pred: ' + str((time_synchronized() - last_time)))

            # Process detections
            for i, det in enumerate(pred_list):  # detections per image
                bfs = buffered_frames_list[i]
                # if  i== len(pred_list)-1:
                #     print(im0s[i].shape, type(im0s[i]))
                #     print("saving")
                #     img_Image = im0s[i].astype(np.uint8)
                #     img_Image = PIL.Image.fromarray(cv2.cvtColor(img_Image, cv2.COLOR_BGR2RGB))
                #     save_dir = os.path.join('./runs/temp2', str(frame_count).zfill(4) + '.jpg')
                #     img_Image.save(save_dir)
                bfs.insert_img((im0s[i], det, None))

            time_cost = (time_synchronized() - start_time)
            logger.debug('time spend :' + str(time_cost))

            # time_sem.acquire()


            logger.info('fps' + str(1 / (time_synchronized() - last_time)))
            logger.info('avg fps: ' + str(frame_count / (time_synchronized() - process_start)))

            last_time = time_synchronized()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='weights/helmet_head_person_m.pt',
                        help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='./src.txt',
                        help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640,
                        help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float,
                        default=0.5, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float,
                        default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true',
                        help='display results')
    parser.add_argument('--save-txt', action='store_true',
                        help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true',
                        help='save confidences in --save-txt labels')
    parser.add_argument('--classes', nargs='+', type=int,
                        help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true',
                        help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true',
                        help='augmented inference')
    parser.add_argument('--update', action='store_true',
                        help='update all models')
    parser.add_argument('--project', default='runs/detect',
                        help='save results to project/name')
    parser.add_argument('--name', default='exp',
                        help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true',
                        help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    print(opt)
    check_requirements()

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
