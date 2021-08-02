# Copyright (c) 2021 Qualcomm Technologies, Inc.

# All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from apex import amp
from runx.logx import logx
import numpy as np
import torch
import argparse
import os
import sys
import time
import fire
from utils.config import assert_and_infer_cfg, cfg
from utils.misc import AverageMeter, eval_metrics
from utils.misc import ImageDumper
from utils.trnval_utils import eval_minibatch
from utils.progress_bar import printProgressBar
from models.loss.utils import get_loss
from models.model_loader import load_model
from library.datasets.get_dataloaders import return_dataloader
import models
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")
    
torch.backends.cudnn.benchmark = True    
    
    
def set_apex_params(local_rank):
    """
    Setting distributed parameters for Apex
    """
    if 'WORLD_SIZE' in os.environ:
        world_size = int(os.environ['WORLD_SIZE'])
        global_rank = int(os.environ['RANK'])
        
    print('GPU {} has Rank {}'.format(
        local_rank, global_rank))
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(backend='nccl',
                                         init_method='env://')
    return world_size, global_rank
       

def inference(val_loader, net, arch, loss_fn, epoch, calc_metrics=True):
    """
    Inference over dataloader on network
    """

    len_dataset = len(val_loader)
    net.eval()
    val_loss = AverageMeter()
    iou_acc = 0

    for val_idx, data in enumerate(val_loader):
        input_images, labels, edge, img_names, _ = data 
        
        # Run network
        assets, _iou_acc = \
            eval_minibatch(data, net, loss_fn, val_loss, calc_metrics,
                          val_idx)
        iou_acc += _iou_acc
        if val_idx+1 < len_dataset:
            printProgressBar(val_idx + 1, len_dataset, 'Progress')    
    
    logx.msg("\n")
    if calc_metrics:
        eval_metrics(iou_acc, net, val_loss, epoch, arch)


def main(output_dir, model_path, has_edge=False, model_summary=False, arch='ocrnet.AuxHRNet', 
         hrnet_base='18', num_workers=4, split='val', batch_size=2, crop_size='1024,2048', 
         apex=True, syncbn=True, fp16=True, local_rank=0):

    #Distributed processing
    if apex:
        world_size, global_rank = set_apex_params(local_rank)
    else:
        world_size = 1
        global_rank = 0  
        local_rank = 0  
        
    #Logging
    logx.initialize(logdir=output_dir,
                    tensorboard=True,
                    global_rank=global_rank)

    #Build config
    assert_and_infer_cfg(output_dir, global_rank, apex, syncbn, arch, hrnet_base,
                         fp16, has_edge)
    
    #Dataloader
    val_loader = return_dataloader(num_workers, batch_size)
    
    #Loss function
    loss_fn = get_loss(has_edge)

    assert model_path is not None, 'need pytorch model for inference'
    
    #Load Network
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    logx.msg("Loading weights from: {}".format(model_path))
    net = models.get_net(arch, loss_fn)
    if fp16:
        net = amp.initialize(net, opt_level='O1', verbosity=0)
    net = models.wrap_network_in_dataparallel(net, apex)
    #restore_net(net, checkpoint, arch)
    load_model(net, checkpoint)
    #Summary of MAC/#param
    if model_summary:
        from thop import profile
        img = torch.randn(1, 3, 1024, 2048).cuda()
        mask = torch.randn(1, 1, 1024, 2048).cuda()
        macs, params = profile(net, inputs=({'images': img, 'gts': mask}, ))
        print(f'macs {macs} params {params}')
        sys.exit()

    
    torch.cuda.empty_cache()
    
    #Run inference
    inference(val_loader, net, arch, loss_fn, epoch=0)


if __name__ == '__main__':
    fire.Fire(main)