import os
import torch


class Options():
    """docstring for Options"""

    def __init__(self):
        pass

    def init(self, parser):
        # global settings
        parser.add_argument('--batch_size', type=int, default=8, help='batch size')
        parser.add_argument('--nepoch', type=int, default=500, help='training epochs')
        parser.add_argument('--train_workers', type=int, default=8, help='train_dataloader workers')
        parser.add_argument('--eval_workers', type=int, default=8, help='eval_dataloader workers')
        parser.add_argument('--optimizer', type=str, default='adamw', help='optimizer for training')
        parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam & adamw')
        parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for adam & adamw')
        parser.add_argument('--lr_initial', type=float, default=3e-4, help='initial learning rate')
        parser.add_argument('--lr_end', type=float, default=1e-6, help='CosineAnnealingLR end learning rate')
        parser.add_argument('--weight_decay', type=float, default=1e-2, help='weight decay')
        parser.add_argument('--gpu', type=str, default='0', help='GPUs')

        # args for saving
        parser.add_argument('--save_dir', type=str, default='../logs/', help='save dir')
        parser.add_argument('--dataset', type=str, default='',
                            help='dataset name, Oracle, WSC41K_mix_128')
        parser.add_argument('--task', type=str, default='',
                            help='task name,')
        parser.add_argument('--save_images', action='store_true', default=False, help='if save images')

        # args for Model
        parser.add_argument('--arch', type=str, default='LWTformer', help='archtechture, LWTformer, LWTformer-B, LWTformer-L')
        parser.add_argument('--dim', type=int, default=32, help='dim of emdeding features')
        parser.add_argument('--loss_type', type=str, default='CharFreqPerceptualLoss',
                            help='loss type for training, CharbonnierLoss, CharFreqLoss, PSNRLoss, FreqLoss, CharFreqPerceptualLoss, Stripformer_Loss, L1Loss, CompositeLoss,CharPerceptualLoss, CharFreqLoss')

        # args for training
        parser.add_argument('--train_ps', type=int, default=128, help='patch size of training sample')
        parser.add_argument('--val_ps', type=int, default=128, help='patch size of validation sample')
        parser.add_argument('--resume', action='store_true', default=False)
        parser.add_argument('--pretrain_weights', type=str,
                            default='',
                            help='path of pretrained_weights')
        parser.add_argument('--train_dir', type=str, default='',
                            help='dir of train data')
        parser.add_argument('--val_dir', type=str, default='',
                            help='dir of test data')
        parser.add_argument('--warmup', action='store_true', default=True, help='warmup')
        parser.add_argument('--warmup_epochs', type=int, default=3, help='epochs for warmup')

        # ddp
        parser.add_argument("--local_rank", type=int, default=-1,
                            help='DDP parameter, do not modify')
        parser.add_argument("--distribute", action='store_true', help='whether using multi gpu train')
        parser.add_argument("--distribute_mode", type=str, default='DDP', help="using which mode to ")
        return parser
