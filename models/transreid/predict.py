import os
import sys
sys.path.append("..")
import torch
from models.transreid.config import cfg
import argparse
import copy
from models.transreid.datasets import make_dataloader
from models.transreid.model import make_model
from models.transreid.processor import do_predict
from models.transreid.utils.logger import setup_logger
from models.transreid.utils.metrics import R1_mAP_eval
import pickle

class Predict:
    def __init__(self, file_name='pickle.pkl', re_compute=True):
        description = "ReID Baseline Training"
        config_file = "models/transreid/configs/DukeMTMC/vit_jpm.yml"
        if config_file != "":
            cfg.merge_from_file(config_file)
        cfg.MODEL.DEVICE_ID = "('0')"
        cfg.TEST.WEIGHT = "models/transreid/weight/duke_vit_transreid_overlap/transformer_60.pth"
        cfg.freeze()
        output_dir = cfg.OUTPUT_DIR
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID

        self.train_loader, self.train_loader_normal, self.val_loader, \
        self.num_query, self.num_classes, self.camera_num, self.view_num = make_dataloader(cfg)

        self.model = make_model(cfg, num_class=self.num_classes, camera_num=self.camera_num, view_num=self.view_num)
        self.model.load_param(cfg.TEST.WEIGHT)
        self.device = "cuda"
        if self.device:
            self.model.to(self.device)
        self.model.eval()
        self.feature_gallery = []
        self.img_path_list = []
        self.evaluator = R1_mAP_eval(self.num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
        self.gallery_labels = []
        if not os.path.exists(file_name) or re_compute:
            print('computing')
            self.compute_feature(file_name=file_name)
        else:
            with open(file_name, 'rb') as f:
                self.feature_gallery = pickle.load(f)
            
        #self.compute_feature(file_name=file_name)
        for n_iter, (img, pid, camid, camids, target_view, imgpath) in enumerate(self.val_loader):
            with torch.no_grad():
                feat = self.feature_gallery[n_iter]
                self.evaluator.update((feat, pid, camid))
                self.img_path_list.extend(imgpath)
                self.gallery_labels.append(imgpath[0][-20:-16])

    def compute_feature(self, file_name):
        for n_iter, (img, pid, camid, camids, target_view, imgpath) in enumerate(self.val_loader):
            with torch.no_grad():
                img = img.to(self.device)
                camids = camids.to(self.device)
                target_view = target_view.to(self.device)
                feat = self.model(img, cam_label=camids, view_label=target_view)
                self.feature_gallery.append(feat)
        with open(file_name, 'wb') as f:
            pickle.dump(self.feature_gallery, f)
    def get_evaluator(self):
        evaluator = copy.deepcopy(self.evaluator)
        return evaluator

    def predict_from_image(self, img_list, evaluator):
        return do_predict(cfg,
                    self.model,
                    self.val_loader,
                    self.num_query,
                    img_list,
                    self.feature_gallery,
                    evaluator,
                    self.device,
                    self.img_path_list,
                    self.gallery_labels,
                   )


