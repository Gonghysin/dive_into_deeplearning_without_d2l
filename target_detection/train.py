import time
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
# from model import tiny_detector, MultiBoxLoss
from datasets import PascalVOCDataset
from modules import data_loader
from utils import *


keep_difficult = True  # use objects considered difficult to detect?
batch_size = 32  # batch size
workers = 4  # number of workers for loading data in the DataLoader

def main():

    # Custom dataloader
    train_dataset = PascalVOCDataset(
        data_loader,
        split='train',
        keep_difficult = keep_difficult
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
        num_workers=workers,
        pin_memory=True
    )