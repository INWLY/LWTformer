import os
import sys

from torch.utils import tensorboard

# Add directories to path
dir_name = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(dir_name, '../dataset/'))
sys.path.append(os.path.join(dir_name, '..'))
print(sys.path)
print(dir_name)

import argparse
from options import train_LWTformer_options

######### Parser ###########
opt = train_LWTformer_options.Options().init(argparse.ArgumentParser(description='Image denoising')).parse_args()
print(opt)

import utils
from dataset.dataset_denoise import *

######### Set GPUs ###########
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
import torch

torch.backends.cudnn.benchmark = True

import torch.optim as optim
from torch.utils.data import DataLoader
import random
import time
import numpy as np
import datetime

from losses import CharFreqPerceptualLoss

from tqdm import tqdm
from warmup_scheduler import GradualWarmupScheduler
from torch.optim.lr_scheduler import StepLR
from timm.utils import NativeScaler

######### Logs Directory ###########
log_dir = os.path.join(opt.save_dir, opt.dataset, opt.task, 'LWTformer')
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
logname = os.path.join(log_dir, datetime.datetime.now().isoformat() + '.txt')
print("Current time: ", datetime.datetime.now().isoformat())
result_dir = os.path.join(log_dir, 'results')
model_dir = os.path.join(log_dir, 'models')
tb_log_dir = os.path.join(log_dir, 'tb_log')
utils.mkdir(result_dir)
utils.mkdir(model_dir)
utils.mkdir(tb_log_dir)
tb_writer = tensorboard.SummaryWriter(log_dir=tb_log_dir)

# ######### Set Seeds ###########
random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)

######### Model ###########
model_restoration = utils.get_arch(opt)

with open(logname, 'a') as f:
    f.write(str(opt) + '\n')

######### Optimizer ###########
start_epoch = 1
if opt.optimizer.lower() == 'adam':
    optimizer = optim.Adam(model_restoration.parameters(), lr=opt.lr_initial, betas=(opt.beta1, opt.beta2), eps=1e-8,
                           weight_decay=opt.weight_decay)
    print(f"Adam initialized, lr: {opt.lr_initial}, betas: {opt.beta1, opt.beta2}, eps: {1e-8}, weight_decay: {opt.weight_decay}")

elif opt.optimizer.lower() == 'adamw':
    optimizer = optim.AdamW(model_restoration.parameters(), lr=opt.lr_initial, betas=(opt.beta1, opt.beta2), eps=1e-8,
                            weight_decay=opt.weight_decay)
    print(f"AdamW initialized, lr: {opt.lr_initial}, betas: {opt.beta1, opt.beta2}, eps: {1e-8}, weight_decay: {opt.weight_decay}")

else:
    raise Exception("Error: Unknown optimizer...")

# Write to log file
with open(logname, 'a') as f:
    f.write(
        f"{opt.optimizer.lower()} initialized, lr: {opt.lr_initial}, betas: {opt.beta1, opt.beta2}, eps: {1e-8}, weight_decay: {opt.weight_decay}" + '\n')

######### DataParallel ###########
model_restoration = torch.nn.DataParallel(model_restoration)
model_restoration.cuda()

######### Scheduler ###########
if opt.warmup:
    print("Using warmup and cosine strategy! Learning rate: {} -> {}".format(opt.lr_initial, opt.lr_end))
    # Write to log file
    with open(logname, 'a') as f:
        f.write(
            "Using warmup and cosine strategy! Learning rate: {} -> {}".format(opt.lr_initial, opt.lr_end) + '\n')
    warmup_epochs = opt.warmup_epochs
    scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.nepoch - warmup_epochs, eta_min=opt.lr_end)    # Cosine annealing learning rate setup
    scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs,
                                       after_scheduler=scheduler_cosine)
    scheduler.step()
else:
    step = 50
    print("Using StepLR, step size = {}!".format(step))
    scheduler = StepLR(optimizer, step_size=step, gamma=0.5)
    scheduler.step()

# Log model architecture
with open(logname, 'a') as f:
    f.write(str(model_restoration) + '\n')

######### Resume Training ###########
if opt.resume:
    path_chk_rest = opt.pretrain_weights
    print("Resuming from " + path_chk_rest)
    utils.load_checkpoint(model_restoration, path_chk_rest)
    start_epoch = utils.load_start_epoch(path_chk_rest) + 1
    lr = utils.load_optim(optimizer, path_chk_rest)

    # for p in optimizer.param_groups: p['lr'] = lr
    # warmup = False
    # new_lr = lr
    # print('------------------------------------------------------------------------------')
    # print("==> Resuming Training with learning rate:", new_lr)
    # print('------------------------------------------------------------------------------')
    for i in range(1, start_epoch):
        scheduler.step()
    new_lr = scheduler.get_lr()[0]
    print('------------------------------------------------------------------------------')
    print("==> Resuming Training with learning rate:", new_lr)
    print('------------------------------------------------------------------------------')

    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.nepoch-start_epoch+1, eta_min=1e-6)

######### Loss Function ###########
if opt.loss_type == 'CharFreqPerceptualLoss':
    criterion = CharFreqPerceptualLoss().cuda()
    print('CharFreqPerceptualLoss initialized')

else:
    raise Exception("Error: Unknown loss_type")

######### DataLoader ###########
print('===> Loading datasets')
img_options_train = {'patch_size': opt.train_ps}
train_dataset = get_training_data(opt.train_dir, img_options_train)
train_loader = DataLoader(dataset=train_dataset, batch_size=opt.batch_size, shuffle=True,
                          num_workers=opt.train_workers, pin_memory=False, drop_last=False)
val_dataset = get_validation_data(opt.val_dir)
val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False,
                        num_workers=opt.eval_workers, pin_memory=False, drop_last=False)

len_trainset = train_dataset.__len__()
len_valset = val_dataset.__len__()
print("Size of training set: ", len_trainset, ", size of validation set: ", len_valset)

######### Training ###########
print('===> Start Epoch {} End Epoch {}'.format(start_epoch, opt.nepoch))

loss_scaler = NativeScaler()
torch.cuda.empty_cache()
for epoch in range(start_epoch, opt.nepoch + 1):
    epoch_start_time = time.time()
    epoch_loss = 0
    train_id = 1

    train_loader_tqdm = tqdm(train_loader, desc=f'Epoch {epoch}/{opt.nepoch}')
    for i, data in enumerate(train_loader_tqdm, 0):
        # Zero gradients
        optimizer.zero_grad()

        target = data[0].cuda()
        input_ = data[1].cuda()

        if epoch > 5:
            target, input_ = utils.MixUp_AUG().aug(target, input_)
        with torch.amp.autocast(device_type='cuda'):
            restored = model_restoration(input_)
            loss = criterion(restored, target)
        loss_scaler(
            loss, optimizer, parameters=model_restoration.parameters())
        epoch_loss += loss.item()

    scheduler.step()

    print("------------------------------------------------------------------")
    print("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.7f}".format(
        epoch, time.time() - epoch_start_time, epoch_loss, scheduler.get_lr()[0]))
    tb_writer.add_scalar('Loss', epoch_loss, epoch)
    tb_writer.add_scalar('LearningRate', scheduler.get_lr()[0], epoch)
    print("------------------------------------------------------------------")
    with open(logname, 'a') as f:
        f.write("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.7f}".format(
            epoch, time.time() - epoch_start_time, epoch_loss, scheduler.get_lr()[0]) + '\n')

    torch.save({'epoch': epoch,
                'state_dict': model_restoration.state_dict(),
                'optimizer': optimizer.state_dict()
                }, os.path.join(model_dir, "model_latest.pth"))

    # if epoch % 20 == 0:
    #     torch.save({'epoch': epoch,
    #                 'state_dict': model_restoration.state_dict(),
    #                 'optimizer': optimizer.state_dict()
    #                 }, os.path.join(model_dir, "model_epoch_{}.pth".format(epoch)))

print("Current time: ", datetime.datetime.now().isoformat())