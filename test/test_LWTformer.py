import os
import sys
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from pytorch_fid.fid_score import calculate_fid_given_paths  # Import pytorch-fid

from utils.metrics import calculate_batch_psnr_ssim, calculate_batch_psnr_ssim_color, calculate_batch_lpips

# Add paths
dir_name = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(dir_name, '../dataset/'))
sys.path.append(os.path.join(dir_name, '..'))

# Import other modules
from dataset.dataset_denoise import *
import utils

# Parse arguments
parser = argparse.ArgumentParser(description='Image denoising evaluation on Character')
parser.add_argument('--real_dir', default='',
                    type=str, help='Directory of validation images real_dir')

parser.add_argument('--input_dir', default='',
                    type=str, help='Directory of validation images')
parser.add_argument('--weights', default='',
                    type=str, help='Path to weights')
parser.add_argument('--gpus', default='0', type=str, help='CUDA_VISIBLE_DEVICES')
parser.add_argument('--arch', default='LWTformer', type=str, help='Architecture name')
parser.add_argument('--batch_size', default=1, type=int, help='Batch size for dataloader')
parser.add_argument('--save_images', action='store_true', help='Save denoised images in result directory')
parser.add_argument('--dim', type=int, default=32, help='Number of dimensions')
parser.add_argument('--dd_in', type=int, default=3, help='Input channels')
parser.add_argument('--train_ps', type=int, default=128, help='Patch size of training sample')
args = parser.parse_args()

# Set GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

# Load model
model_restoration = utils.get_arch(args)

# Load checkpoint
utils.load_checkpoint(model_restoration, args.weights)
print("===>Testing using weights: ", args.weights)

# Get starting epoch from checkpoint
start_epoch = utils.load_start_epoch(args.weights)
print("===>Loading Epoch:", start_epoch)

# Move model to GPU and set to evaluation mode
model_restoration.cuda()
model_restoration.eval()

# Load validation data
val_dataset = get_validation_data(args.input_dir)
val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8, drop_last=False)

# Get validation set length
len_valset = val_dataset.__len__()

# Define directories for saving generated and real images
result_dir = './results'
fake_dir = os.path.join(result_dir, 'fake_images')  # Generated images directory
os.makedirs(fake_dir, exist_ok=True)


# Evaluation
with torch.no_grad():
    model_restoration.eval()
    psnr_val_rgb = []
    ssim_val_rgb = []

    all_restored = []
    all_filenames = []

    # Generate denoised images
    for ii, data_val in enumerate(tqdm(val_loader, desc='Generate images'), 0):
        target = data_val[0].cuda()
        input_ = data_val[1].cuda()
        filenames = data_val[2]

        # Use automatic mixed precision for inference
        with torch.amp.autocast(device_type='cuda'):
            restored = model_restoration(input_)

        # Clamp output to valid range [0, 1]
        restored = torch.clamp(restored, 0, 1)

        # Save generated images
        for j in range(restored.shape[0]):
            single_restored = restored[j].unsqueeze(0)  # Extract single image
            single_filename = filenames[j]              # Get corresponding filename

            # Save generated image
            utils.save_img(os.path.join(fake_dir, single_filename), single_restored)

# Calculate PSNR and SSIM
psnr, ssim = calculate_batch_psnr_ssim_color(args.real_dir, fake_dir, target_size=(256, 256))

# Calculate LPIPS
lpips = calculate_batch_lpips(args.real_dir, fake_dir, batch_size=args.batch_size)

# Calculate FID
print('======Calculate FID======')
fid = calculate_fid_given_paths([args.real_dir, fake_dir], batch_size=args.batch_size, device='cuda', dims=2048)


# Print all metrics
print("[PSNR: %.4f\t SSIM: %.4f\t LPIPS: %.4f\t FID: %.4f]" % (psnr, ssim, lpips, fid))
