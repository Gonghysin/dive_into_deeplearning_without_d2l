import time
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
# from model import tiny_detector, MultiBoxLoss
from datasets import PascalVOCDataset
from modules import data_loader
from target_detection.model import TinyDetector , MultiBoxLoss
from utils import *

device= torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True

# Data parameters
data_folder = '/root/autodl-tmp/dive_into_deeplearning_without_d2l/data/VOCdevkit'  # data files root path
keep_difficult = True  # use objects considered difficult to detect?
n_classes = len(label_map)  # number of different types of objects


# Learning parameters
total_epochs = 230 # number of epochs to train
batch_size = 32  # batch size
workers = 4  # number of workers for loading data in the DataLoader
print_freq = 100  # print training status every __ batches
lr = 1e-3  # learning rate
decay_lr_at = [150, 190]  # decay learning rate after these many epochs
decay_lr_to = 0.1  # decay learning rate to this fraction of the existing learning rate
momentum = 0.9  # momentum
weight_decay = 5e-4  # weight decay

def train(train_loader, model, criterion, optimizer, epoch):
    """
    One epoch's training.

    :param train_loader: DataLoader for training data
    :param model: model
    :param criterion: MultiBox loss
    :param optimizer: optimizer
    :param epoch: epoch number
    """
    model.train()  # training mode enables dropout

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss

    start = time.time()

    # Batches
    for i, (images, boxes, labels, _) in enumerate(train_loader):
        data_time.update(time.time() - start)

        # Move to default device
        images = images.to(device)  # (batch_size (N), 3, 224, 224)
        boxes = [b.to(device) for b in boxes]
        labels = [l.to(device) for l in labels]

        # Forward prop.
        predicted_locs, predicted_scores = model(images)  # (N, 441, 4), (N, 441, n_classes)

        # Loss
        loss = criterion(predicted_locs, predicted_scores, boxes, labels)  # scalar

        # Backward prop.
        optimizer.zero_grad()
        loss.backward()

        # Update model
        optimizer.step()

        losses.update(loss.item(), images.size(0))
        batch_time.update(time.time() - start)

        start = time.time()

        # Print status
        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(epoch,
                                                                  i, 
                                                                  len(train_loader),
                                                                  batch_time=batch_time,
                                                                  data_time=data_time, 
                                                                  loss=losses))
    del predicted_locs, predicted_scores, images, boxes, labels  # free some memory since their histories may be stored


def main():
    # Initialize model and optimizer
    model = TinyDetector(n_classes=n_classes)
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    criterion = MultiBoxLoss(priors_cxcy=model.priors_cxcy).to(device)


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

    # Epochs
    for epoch in range(total_epochs):
        # Decay learning rate at particular epochs
        if epoch in decay_lr_at:
            adjust_learning_rate(optimizer, decay_lr_to)

        # One epoch's training                                                                                                                 
        train(train_loader=train_loader,
              model=model,
              criterion=criterion,
              optimizer=optimizer,
              epoch=epoch)

        # Save checkpoint
        save_checkpoint(epoch, model, optimizer)