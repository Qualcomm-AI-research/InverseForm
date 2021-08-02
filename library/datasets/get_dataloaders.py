import library.transforms.joint_transforms as joint_transforms
import library.transforms.transforms as extended_transforms
from torch.utils.data import DataLoader

import importlib
import torchvision.transforms as standard_transforms

from utils.config import cfg
from runx.logx import logx


def return_dataloader(num_workers, batch_size):
    """
    Return Dataloader
    """

    base_loader = importlib.import_module('library.datasets.base_loader')
    dataset = getattr(base_loader, cfg.DATASET.NAME)

    val_joint_transform_list = None

    mean_std = (cfg.DATASET.MEAN, cfg.DATASET.STD)
    target_transform = extended_transforms.MaskToTensor()

    val_input_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*mean_std)
    ])

    val_frame = dataset(
        mode='val',
        joint_transform_list=val_joint_transform_list,
        img_transform=val_input_transform,
        label_transform=target_transform,
        eval_folder=None)

    if cfg.apex:
        from library.datasets.sampler import DistributedSampler
        val_sampler = DistributedSampler(val_frame, pad=False, permutation=False,
                                         consecutive_sample=False)
    else:
        val_sampler = None

    val_loader = DataLoader(val_frame, batch_size=batch_size,
                            num_workers=num_workers // 2,
                            shuffle=False, drop_last=False,
                            sampler=val_sampler)

    return val_loader
