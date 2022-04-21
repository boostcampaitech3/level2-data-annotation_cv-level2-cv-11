import os
import os.path as osp
import time
import math
from datetime import timedelta
from argparse import ArgumentParser

import torch
import numpy as np
from torch import cuda
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from tqdm import tqdm

from deteval import calc_deteval_metrics
from detect import detect, get_bboxes
from east_dataset import EASTDataset
from dataset import SceneTextDataset
from model import EAST
import wandb


def parse_args():
    parser = ArgumentParser()

    # Conventional args
    parser.add_argument('--data_dir', type=str,
                        default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data/total'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR',
                                                                        'trained_models'))

    parser.add_argument('--device', default='cuda' if cuda.is_available() else 'cpu')
    parser.add_argument('--num_workers', type=int, default=8)

    parser.add_argument('--image_size', type=int, default=1024)
    parser.add_argument('--input_size', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--max_epoch', type=int, default=200)
    parser.add_argument('--save_interval', type=int, default=5)
    parser.add_argument('--wandb_plot', type=bool, default=True)
    parser.add_argument('--validate', type=bool, default=False)
    parser.add_argument('--val_interval', type=int, default=5, help='validate per n(default=5) epochs')
    parser.add_argument('--pretrained_path', default = '/opt/ml/input/data/models/trained_models_icdar19/epoch_50.pth')
    args = parser.parse_args()

    if args.input_size % 32 != 0:
        raise ValueError('`input_size` must be a multiple of 32')

    return args


def do_training(data_dir, model_dir, device, image_size, input_size, num_workers, batch_size,
                learning_rate, max_epoch, save_interval, wandb_plot, validate, val_interval, pretrained_path):
    
    # wandb init
    if wandb_plot:
        wandb.init(project="ocr-model", entity="canvas11", name = "LEE-total-batch32-aug")

    train_dataset = SceneTextDataset(data_dir, split='icdar19_and_total_11146', image_size=image_size, crop_size=input_size) # split에 val의 [].json의 [] 부분을 입력.
    train_dataset = EASTDataset(train_dataset)
    train_num_batches = math.ceil(len(train_dataset) / batch_size)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    if validate:
        val_dataset = SceneTextDataset(data_dir, split='cv_stratified_val', image_size=image_size, crop_size=input_size)
        val_dataset = EASTDataset(val_dataset)
        val_num_batches = math.ceil(len(val_dataset) / batch_size)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = EAST()
    model.load_state_dict(torch.load(pretrained_path))
    print('loaded')
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[max_epoch // 4, max_epoch // 2], gamma=0.1)
    # scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor = 0.1, patience=10, verbose=True)

    for epoch in range(max_epoch):
        # train loop 
        model.train()
        epoch_loss, epoch_start = 0, time.time()
        cls_loss, angle_loss, iou_loss = 0, 0, 0
        with tqdm(total=train_num_batches) as pbar:
            for img, gt_score_map, gt_geo_map, roi_mask in train_loader:
                pbar.set_description('[Epoch {}]'.format(epoch + 1))

                loss, extra_info, _, _ = model.train_step(img.to(device), gt_score_map, gt_geo_map, roi_mask)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_val = loss.item()
                epoch_loss += loss_val

                pbar.update(1)
                val_dict = {
                    'Cls loss': extra_info['cls_loss'], 'Angle loss': extra_info['angle_loss'],
                    'IoU loss': extra_info['iou_loss']
                }

                cls_loss += extra_info['cls_loss']
                angle_loss += extra_info['angle_loss']
                iou_loss += extra_info['iou_loss']

                pbar.set_postfix(val_dict)
       
        print('Mean loss: {:.4f} | Elapsed time: {}'.format(
            epoch_loss / train_num_batches, timedelta(seconds=time.time() - epoch_start)))

        if wandb_plot:
            wandb.log({'Train/Mean loss': epoch_loss / train_num_batches,
                    'Train/Cls loss': cls_loss/train_num_batches, 
                    'Train/Angle loss': angle_loss/train_num_batches,
                    'Train/IoU loss': iou_loss/train_num_batches})

        if validate and (epoch+1) % val_interval == 0:
            # validate loop
            model.eval()
            gt_bboxes, pred_bboxes, trans = [], [], []
            orig_sizes = []
            epoch_loss, epoch_start = 0, time.time()
            cls_loss, angle_loss, iou_loss = 0, 0, 0

            with tqdm(total=val_num_batches) as vbar:
                for img, gt_score_map, gt_geo_map, roi_mask in val_loader:
                    vbar.set_description('[Validate]')
                    for image in img:
                        orig_sizes.append(image.shape[1:3])
                    gt_bbox = []
                    pred_bbox = []
                    tran = []
                    with torch.no_grad():
                        # pred_score_map, pred_geo_map = model.forward(img.to(device))
                        loss, extra_info, pred_score_map, pred_geo_map = model.train_step(img.to(device), gt_score_map, gt_geo_map, roi_mask)
                    
                    loss_val = loss.item()
                    epoch_loss += loss_val
                    val_dict = {
                        'Cls loss': extra_info['cls_loss'], 'Angle loss': extra_info['angle_loss'],
                        'IoU loss': extra_info['iou_loss']
                    }

                    cls_loss += extra_info['cls_loss']
                    angle_loss += extra_info['angle_loss']
                    iou_loss += extra_info['iou_loss']
                    
                    vbar.set_postfix(val_dict)


                    for gt_score, gt_geo, pred_score, pred_geo, orig_size in zip(gt_score_map.cpu().numpy(), gt_geo_map.cpu().numpy(), pred_score_map.cpu().numpy(), pred_geo_map.cpu().numpy(), orig_sizes):
                        gt_bbox_angle = get_bboxes(gt_score, gt_geo)
                        pred_bbox_angle = get_bboxes(pred_score, pred_geo)
                        if gt_bbox_angle is None:
                            gt_bbox_angle = np.zeros((0, 4, 2), dtype=np.float32)
                            tran_angle = []
                        else:
                            gt_bbox_angle = gt_bbox_angle[:, :8].reshape(-1, 4, 2)
                            gt_bbox_angle *= max(orig_size) / input_size
                            tran_angle = ['null' for _ in range(gt_bbox_angle.shape[0])]
                        if pred_bbox_angle is None:
                            pred_bbox_angle = np.zeros((0, 4, 2), dtype=np.float32)
                        else:
                            pred_bbox_angle = pred_bbox_angle[:, :8].reshape(-1, 4, 2)
                            pred_bbox_angle *= max(orig_size) / input_size
                            
                        tran.append(tran_angle)
                        gt_bbox.append(gt_bbox_angle)
                        pred_bbox.append(pred_bbox_angle)
                    
                    vbar.update(1)
                    gt_bboxes.extend(gt_bbox)
                    pred_bboxes.extend(pred_bbox)
                    trans.extend(tran)

            img_len = len(val_dataset)
            pred_bboxes_dict, gt_bboxes_dict, trans_dict = dict(), dict(), dict()
            for img_num in range(img_len):
                pred_bboxes_dict[f'img_{img_num}'] = pred_bboxes[img_num]
                gt_bboxes_dict[f'img_{img_num}'] = gt_bboxes[img_num]
                trans_dict[f'img_{img_num}'] = trans[img_num]
            
            deteval_dict = calc_deteval_metrics(pred_bboxes_dict, gt_bboxes_dict, trans_dict)
            metric_dict = deteval_dict['total']
            precision = metric_dict['precision']
            recall = metric_dict['recall']
            hmean = metric_dict['hmean']
            print(f"(Validate) Precision: {round(precision,4)}, Recall: {round(recall,4)}, Hmean: {round(hmean,4)}")

            if wandb_plot:
                wandb.log({'Val/Precision': precision,
                        'Val/Recall': recall, 
                        'Val/Hmean': hmean,
                        'Val/Mean loss': epoch_loss / val_num_batches,
                        'Val/Cls loss': cls_loss/val_num_batches, 
                        'Val/Angle loss': angle_loss/val_num_batches,
                        'Val/IoU loss': iou_loss/val_num_batches
                        })

        # scheduler.step(epoch_loss / val_num_batches) # Mean Loss 기준
        scheduler.step()
        if (epoch + 1) % save_interval == 0:
            if not osp.exists(model_dir):
                os.makedirs(model_dir)

            ckpt_fpath = osp.join(model_dir, f'epoch_{epoch+1}.pth')
            torch.save(model.state_dict(), ckpt_fpath)


def main(args):
    do_training(**args.__dict__)


if __name__ == '__main__':
    args = parse_args()
    main(args)
