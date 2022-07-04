import logging
import os
import time
import torch
import torch.nn as nn
import sys
sys.path.append("..")
from models.transreid.utils.meter import AverageMeter
from models.transreid.utils.metrics import R1_mAP_eval
from torch.cuda import amp
import torch.distributed as dist
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torchvision.transforms as T
from utils.plots import Colors

from torchvision import transforms
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
def do_train(cfg,
             model,
             center_criterion,
             train_loader,
             val_loader,
             optimizer,
             optimizer_center,
             scheduler,
             loss_fn,
             num_query, local_rank):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD

    device = "cuda"
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("transreid.train")
    logger.info('start training')
    _LOCAL_PROCESS_GROUP = None
    if device:
        model.to(local_rank)
        if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    scaler = amp.GradScaler()
    # train
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()
        evaluator.reset()
        scheduler.step(epoch)
        model.train()
        for n_iter, (img, vid, target_cam, target_view) in enumerate(train_loader):
            optimizer.zero_grad()
            optimizer_center.zero_grad()
            img = img.to(device)
            target = vid.to(device)
            target_cam = target_cam.to(device)
            target_view = target_view.to(device)
            with amp.autocast(enabled=True):
                score, feat = model(img, target, cam_label=target_cam, view_label=target_view )
                loss = loss_fn(score, feat, target, target_cam)

            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()

            if 'center' in cfg.MODEL.METRIC_LOSS_TYPE:
                for param in center_criterion.parameters():
                    param.grad.data *= (1. / cfg.SOLVER.CENTER_LOSS_WEIGHT)
                scaler.step(optimizer_center)
                scaler.update()
            if isinstance(score, list):
                acc = (score[0].max(1)[1] == target).float().mean()
            else:
                acc = (score.max(1)[1] == target).float().mean()

            loss_meter.update(loss.item(), img.shape[0])
            acc_meter.update(acc, 1)

            torch.cuda.synchronize()
            if (n_iter + 1) % log_period == 0:
                logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                            .format(epoch, (n_iter + 1), len(train_loader),
                                    loss_meter.avg, acc_meter.avg, scheduler._get_lr(epoch)[0]))

        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        if cfg.MODEL.DIST_TRAIN:
            pass
        else:
            logger.info("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                    .format(epoch, time_per_batch, train_loader.batch_size / time_per_batch))

        if epoch % checkpoint_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    torch.save(model.state_dict(),
                               os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))
            else:
                torch.save(model.state_dict(),
                           os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))

        if epoch % eval_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    model.eval()
                    for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
                        with torch.no_grad():
                            img = img.to(device)
                            camids = camids.to(device)
                            target_view = target_view.to(device)
                            feat = model(img, cam_label=camids, view_label=target_view)
                            evaluator.update((feat, vid, camid))
                    cmc, mAP, _, _, _, _, _ = evaluator.compute()
                    logger.info("Validation Results - Epoch: {}".format(epoch))
                    logger.info("mAP: {:.1%}".format(mAP))
                    for r in [1, 5, 10]:
                        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                    torch.cuda.empty_cache()
            else:
                model.eval()
                for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
                    with torch.no_grad():
                        img = img.to(device)
                        camids = camids.to(device)
                        target_view = target_view.to(device)
                        feat = model(img, cam_label=camids, view_label=target_view)
                        evaluator.update((feat, vid, camid))
                cmc, mAP, _, _, _, _, _ = evaluator.compute()
                logger.info("Validation Results - Epoch: {}".format(epoch))
                logger.info("mAP: {:.1%}".format(mAP))
                for r in [1, 5, 10]:
                    logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                torch.cuda.empty_cache()


def do_inference(cfg,
                 model,
                 val_loader,
                 num_query):
    device = "cuda"
    logger = logging.getLogger("transreid.test")
    logger.info("Enter inferencing")

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)

    evaluator.reset()

    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    model.eval()
    img_path_list = []

    for n_iter, (img, pid, camid, camids, target_view, imgpath) in enumerate(val_loader):
        with torch.no_grad():
            img = img.to(device)
            camids = camids.to(device)
            target_view = target_view.to(device)
            feat = model(img, cam_label=camids, view_label=target_view)
            evaluator.update((feat, pid, camid))
            img_path_list.extend(imgpath)

    cmc, mAP, _, _, _, _, _ = evaluator.compute()
    logger.info("Validation Results ")
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    return cmc[0], cmc[4]
from utils.torch_utils import select_device, time_synchronized


def do_predict(cfg,
               model,
               val_loader,
               num_query,
               img_list,
               feature_gallery,
               evaluator=R1_mAP_eval(50, max_rank=50, feat_norm='yes'),
               device='cuda',
               img_path_list=[],
               gallery_labels=[],
               add_feature=False,):

    img_names = []
    data_transform = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TEST),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    ])

    # class_name = ['张培基', '余洋', '胡杨柳', '电厂员工1', '电厂员工2', '童琳']
    img_list_t = torch.Tensor(1, 3, 256, 128) #将图片转为可送入model的形式 PIL->Tensor
    for img in img_list:
        ori_img = data_transform(img)
        ori_img = torch.unsqueeze(ori_img, dim=0)
        img_list_t = torch.cat((img_list_t, ori_img), dim=0)
    show_list = img_list
    img_list = img_list_t[1:]
    pic_num = len(img_list)
    with torch.no_grad():
        img_list = img_list.to(device)
        camids = torch.tensor([1], device=device)
        target_view = torch.tensor([0], device=device)
        # t1 = time_sync()
        feat = model(img_list, cam_label=camids, view_label=target_view)
        # t2 = time_sync()
        # print("Transformer time: ", t2 - t1)
        evaluator.update((feat, (0, 0), (0, 0)))

        # t3 = time_sync()
        # print("update time: ", t3 - t2)
        t3 = time_synchronized()
        label_predicts = evaluator.compute(is_pridect=True, num_query=pic_num, add_feature=add_feature, img_path=img_path_list)
        t4 = time_synchronized()
        print('transformer time:', t4-t3)
        evaluator.delete()
        # t4 = time_sync()
        # print("ReID time: ", t4 - t3)
        label_list = []
        for i in range(0, len(label_predicts)):
            result = {}
            for j in range(5):
                #print('label_predicts[i][j]', label_predicts[i][j], 'i', i, 'j', j)
                t = gallery_labels[label_predicts[i][j]]
                if result.get(t) == None:
                    result[t] = 1
                else:
                    result[t] += 1
            max_count = 0
            max_j = ''
            for key in result:
                if result[key] > max_count:
                    max_count = result[key]
                    max_j = key
            label_list.append(max_j)
        #for i, show_img in enumerate(show_list):
            #k = 'predict: ' + label_list[i]
            #plt.text(0.5, 1, k, bbox=dict(boxstyle='round,pad=0.5', fc='yellow', ec='k', lw=1, alpha=0.5))
            # show_img = mpimg.imread(img_path + path)
            # plt.imshow(show_img)
            # plt.show()
            #os.rename(img_path+path, img_path+new_name)
        # print("total time: ", t4-t1)
        t5 = time_synchronized()
        print('metrics time: ', t5-t4)
        print('number of sum person :', pic_num)
        return label_list
