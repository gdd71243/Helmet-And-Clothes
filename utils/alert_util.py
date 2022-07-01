# coding=utf-8
'''
create_data:2021/10/27

'''
import base64
import json
import threading
import time
from time import sleep

import yaml
import requests
import datetime
import cv2
from loguru import logger


def load_conf(path='./strategy_config.yaml'):
    with open(path) as f:
        config = yaml.load(f.read(), Loader=yaml.FullLoader)
        config['show_cam'] = ''
        config['continue_run'] = True
        config['streams'] = {}
        logger.info(config)
        return config


class Strategy_util:
    def __init__(self):

        self.config = load_conf()
        self.LAST_ACK = time.time()

        threading.Thread(target=self.start_heartbeat).start()

    def heartbeat(self, ):
        url = self.config['post_url']

        data = {
            'cam_list': '/'.join(self.parse_cam_info()[0]),
            'strategy_id': self.config['strategy_id'],

        }

        # logger.debug('-----------beat---------')

        res = requests.post(url + '/strategy/heartbeat/', data=data)
        self.config['show_cam'] = res.json()['data']['show_cam']
        self.config['continue_run'] = res.json()['data']['continue_run']
        self.config['threshold'] = res.json()['data']['threshold']
        self.config['sensibility'] = res.json()['data']['sensibility']
        self.config['streams'] = res.json()['data']['streams']
        if not self.config['continue_run']:
            logger.info('------------收到指令停止---------')
        # logger.info(res.json())

        self.LAST_ACK = max(time.time(), self.LAST_ACK)

    def parse_cam_info(self):
        return list(self.config['streams'].keys()), list(self.config['streams'].values())

    def start_heartbeat(self):
        self.LAST_ACK = time.time()
        # url = self.config['post_url']

        while True:

            # threading.Thread(target=heartbeat, args=(url,)).start()
            self.heartbeat()
            if abs(time.time() - self.LAST_ACK) > 3600:
                self.config['continue_run'] = False
            sleep(1.5)
            if not self.config['continue_run']:
                break

    def cam_info(self):
        return self.config['stream']

    def post_alert(self, img, video=None, cam_id=1, level=1):
        url = self.config['post_url']

        logger.info('start post alert')
        data = {
            'level': level,
            'strategy_id': self.config['strategy_id'],
            'cam_id': cam_id,
            "Content-Type": "application/octet-stream",
            "Content-Disposition": "form-data",
        }
        now = str(datetime.datetime.now().timestamp()).replace('.', '')

        file = {
            'snapshot': ('img' + now + '.jpg', img, 'image/jpg'),
            'video': ('last' + now + '.mp4', video, 'video/mp4')
        }

        res = requests.post(url + '/alert/raise_alert/', data=data, files=file)
        res_parse = res.json()
        logger.info(res_parse)
        logger.info('fin post alert')
        return res_parse['alert_id']

    def post_alert_video(self, alert_id, video):
        url = self.config['post_url']

        logger.info('start post alert video')
        data = {
            # TODO 平台v2.0版本需要改接口
            'alert_id': alert_id,
            "Content-Type": "application/octet-stream",
            "Content-Disposition": "form-data",
        }
        now = str(datetime.datetime.now().timestamp()).replace('.', '')

        file = {
            'video': ('last' + now + '.mp4', video, 'video/mp4')
        }

        res = requests.post(url + '/alert/post_video/', data=data, files=file)
        logger.info(res.json())
        logger.info('fin post alert video')

    # def get_CONF(self):
    #     return self.config


IMG_POSTING_BUFFER = []


# def post_real_time_video(video):
#     global self.config
#     url = self.config['post_url']
#
#     logger.info('start post rtv')
#     data = {
#         "Content-Type": "application/octet-stream",
#         "Content-Disposition": "form-data",
#     }
#     now = str(datetime.datetime.now().timestamp()).replace('.', '')
#
#     file = {
#         'video': ('last' + now + '.mp4', video, 'video/mp4')
#     }
#
#     res = requests.post(url + '/monitor/upload_real_time_video/', data=data, files=file)
#     logger.info(res)
#     logger.info('fin post rtv')


def post_real_time_frame(frame, url):
    # logger.info('start post rtf')
    data = {
        "Content-Type": "application/octet-stream",
        "Content-Disposition": "form-data",
    }
    now = str(datetime.datetime.now().timestamp()).replace('.', '')

    file = {
        'frame': ('last' + now + '.jpg', frame, 'image/jpg')
    }

    res = requests.post(url + '/monitor/upload_real_time_frame/', data=data, files=file)
    # logger.info(res.json())
    # logger.info('fin post rtf')


# 当Buffer中有frame时发给服务器
def send_frames():
    config = load_conf()
    url = config['post_url']

    while True:
        while IMG_POSTING_BUFFER:
            try:
                frame = IMG_POSTING_BUFFER.pop()
                success, encode_frame = cv2.imencode(".jpg", frame)
                encode_bytes = encode_frame.tobytes()

                post_real_time_frame(encode_bytes, url=url)
            except Exception as e:
                logger.info(e)
        time.sleep(0.02)


threading.Thread(target=send_frames).start()
