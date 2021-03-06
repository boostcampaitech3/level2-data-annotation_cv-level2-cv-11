import os
import os.path as osp
from random import shuffle
import time
import math
from datetime import timedelta
from argparse import ArgumentParser

import torch
from torch import cuda
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from tqdm import tqdm

from east_dataset import EASTDataset
from dataset import SceneTextDataset
from model import EAST
import wandb


def parse_args():
    parser = ArgumentParser()

    # Conventional args
    parser.add_argument('--data_dir', type=str,
                        default=os.environ.get('SM_CHANNEL_TRAIN', '../input/data/ICDAR17_Korean'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR',
                                                                        'trained_models'))

    parser.add_argument('--device', default='cuda' if cuda.is_available() else 'cpu')
    parser.add_argument('--num_workers', type=int, default=8)

    parser.add_argument('--image_size', type=int, default=1024)
    parser.add_argument('--input_size', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--max_epoch', type=int, default=200)
    parser.add_argument('--save_interval', type=int, default=5)
    parser.add_argument('--wandb_plot', type=bool, default=False)
    parser.add_argument('--validate', type=bool, default=False)
    parser.add_argument('--val_interval', type=int, default=5, help='validate per n(default=5) epochs')


    args = parser.parse_args()

    if args.input_size % 32 != 0:
        raise ValueError('`input_size` must be a multiple of 32')

    return args


def do_training(data_dir, model_dir, device, image_size, input_size, num_workers, batch_size,
                learning_rate, max_epoch, save_interval, wandb_plot, validate, val_interval):
    
    # wandb init
    if wandb_plot:
        wandb.init(project="ocr-model", entity="canvas11", name = "NAME")

    train_dataset = SceneTextDataset(data_dir, split='train', image_size=image_size, crop_size=input_size) # split??? train??? [].json??? [] ????????? ??????.
    train_dataset = EASTDataset(train_dataset)
    train_num_batches = math.ceil(len(train_dataset) / batch_size)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    if validate:
        val_dataset = SceneTextDataset(data_dir, split='train', image_size=image_size, crop_size=input_size)# split??? val??? [].json??? [] ????????? ??????.
        val_dataset = EASTDataset(val_dataset)
        val_num_batches = math.ceil(len(val_dataset) / batch_size)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = EAST()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[max_epoch // 2], gamma=0.1) # Multi step
    # scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_num_batches) # Cosine Annealing

    for epoch in range(max_epoch):
        # train loop
        model.train()
        epoch_loss, epoch_start = 0, time.time()
        cls_loss, angle_loss, iou_loss = 0, 0, 0
        with tqdm(total=train_num_batches) as pbar:
            for img, gt_score_map, gt_geo_map, roi_mask in train_loader:
                pbar.set_description('[Epoch {}]'.format(epoch + 1))

                loss, extra_info = model.train_step(img, gt_score_map, gt_geo_map, roi_mask)
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
            with torch.no_grad():
                epoch_loss, epoch_start = 0, time.time()
                cls_loss, angle_loss, iou_loss = 0, 0, 0
                with tqdm(total=val_num_batches) as vbar:
                    for img, gt_score_map, gt_geo_map, roi_mask in val_loader:
                        vbar.set_description('[Validatae]')

                        loss, extra_info = model.train_step(img, gt_score_map, gt_geo_map, roi_mask)
                        
                        loss_val = loss.item()
                        epoch_loss += loss_val

                        vbar.update(1)
                        val_dict = {
                            'Cls loss': extra_info['cls_loss'], 'Angle loss': extra_info['angle_loss'],
                            'IoU loss': extra_info['iou_loss']
                        }

                        cls_loss += extra_info['cls_loss']
                        angle_loss += extra_info['angle_loss']
                        iou_loss += extra_info['iou_loss']

                        vbar.set_postfix(val_dict)
            
            print('Val Mean loss: {:.4f} | Elapsed time: {}'.format(
                epoch_loss / val_num_batches, timedelta(seconds=time.time() - epoch_start)))

            if wandb_plot:
                wandb.log({'Val/Mean loss': epoch_loss / val_num_batches,
                        'Val/Cls loss': cls_loss/val_num_batches, 
                        'Val/Angle loss': angle_loss/val_num_batches,
                        'Val/IoU loss': iou_loss/val_num_batches})

        scheduler.step()

        if (epoch + 1) % save_interval == 0:
            if not osp.exists(model_dir):
                os.makedirs(model_dir)

            ckpt_fpath = osp.join(model_dir, 'latest.pth')
            torch.save(model.state_dict(), ckpt_fpath)


def main(args):
    do_training(**args.__dict__)


if __name__ == '__main__':
    args = parse_args()
    main(args)
