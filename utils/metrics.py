import os

######### Set GPUs ###########
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import numpy as np
import torch
import lpips
import cv2
from pytorch_fid.fid_score import calculate_fid_given_paths

from torchvision import transforms
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity
from PIL import Image

from tqdm import tqdm


def calculate_batch_lpips(real_dir, fake_dir, net='vgg', device='cuda', batch_size=16):
    """
    Optimized LPIPS calculation function with batch processing support

    Args:
        real_dir: Directory containing real images
        fake_dir: Directory containing generated images
        net: Backbone network ('alex'|'vgg'|'squeeze')
        device: Computation device
        batch_size: Batch size for processing

    Returns:
        Mean LPIPS value
    """
    # Initialize model
    loss_fn = lpips.LPIPS(net=net).to(device).eval()
    for param in loss_fn.parameters():
        param.requires_grad = False

    # Image preprocessing
    transform = transforms.Compose([
        transforms.Resize(256),  # Resize to recommended size
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Load and validate image paths
    real_paths = sorted([os.path.join(real_dir, f) for f in os.listdir(real_dir)
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    fake_paths = sorted([os.path.join(fake_dir, f) for f in os.listdir(fake_dir)
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    assert len(real_paths) == len(fake_paths), "Mismatched number of files"
    assert all(os.path.basename(r) == os.path.basename(f)
               for r, f in zip(real_paths, fake_paths)), "Filename mismatch"

    # Batch processing
    lpips_values = []
    for i in tqdm(range(0, len(real_paths), batch_size),
                  desc=f"LPIPS ({net})", unit="batch"):
        # Load batch
        real_batch = torch.stack([
            transform(Image.open(p).convert('RGB'))
            for p in real_paths[i:i + batch_size]
        ]).to(device)

        fake_batch = torch.stack([
            transform(Image.open(p).convert('RGB'))
            for p in fake_paths[i:i + batch_size]
        ]).to(device)

        # Calculate LPIPS
        with torch.no_grad():
            batch_lpips = loss_fn(real_batch, fake_batch)
            lpips_values.extend(batch_lpips.cpu().numpy())

    return np.mean(lpips_values)


def calculate_batch_psnr_ssim(real_dir, fake_dir, data_range=255.0, target_size=(128, 128)):
    """
    Calculate PSNR and SSIM using OpenCV

    Args:
        real_dir: Directory containing reference images
        fake_dir: Directory containing generated images
        data_range: Pixel value range (255 for [0,255], 1 for [0,1])
        target_size: Target image size (width, height)

    Returns:
        Tuple of (mean PSNR, mean SSIM)
    """
    # Get matched file lists
    real_files = sorted([f for f in os.listdir(real_dir)
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    fake_files = sorted([f for f in os.listdir(fake_dir)
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    assert len(real_files) == len(fake_files), "Mismatched number of files"
    assert real_files == fake_files, "Filename mismatch"

    psnr_values = []
    ssim_values = []
    error_files = []

    for filename in tqdm(real_files, desc="Calculate PSNR & SSIM"):
        try:
            # Load images with OpenCV (BGR format)
            real_img = cv2.imread(os.path.join(real_dir, filename), cv2.IMREAD_COLOR)
            fake_img = cv2.imread(os.path.join(fake_dir, filename), cv2.IMREAD_COLOR)

            # Validate loading
            if real_img is None or fake_img is None:
                raise ValueError("Image loading failed")

            # Resize to target dimensions
            real_img = cv2.resize(real_img, target_size, interpolation=cv2.INTER_AREA)
            fake_img = cv2.resize(fake_img, target_size, interpolation=cv2.INTER_AREA)

            # Convert BGR to RGB
            real_img = cv2.cvtColor(real_img, cv2.COLOR_BGR2RGB)
            fake_img = cv2.cvtColor(fake_img, cv2.COLOR_BGR2RGB)

            # Convert to float
            real_img = real_img.astype(np.float64)
            fake_img = fake_img.astype(np.float64)

            # Convert to grayscale using OpenCV's conversion
            real_gray = cv2.cvtColor(real_img.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float64)
            fake_gray = cv2.cvtColor(fake_img.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float64)

            # Calculate metrics
            psnr = peak_signal_noise_ratio(real_gray, fake_gray, data_range=data_range)
            ssim = structural_similarity(real_gray, fake_gray, data_range=data_range)

            psnr_values.append(psnr)
            ssim_values.append(ssim)

        except Exception as e:
            error_files.append(filename)
            print(f"\nProcessing failed: {filename} - Error: {str(e)}")
            continue

    if error_files:
        print(f"\nWarning: {len(error_files)} files failed processing")
        with open("error_log.txt", "w") as f:
            f.write("\n".join(error_files))

    return np.mean(psnr_values), np.mean(ssim_values)


def calculate_batch_psnr_ssim_pt(pred_tensor, target_tensor, data_range=1.0, average=False):
    """
    Calculate PSNR and SSIM for tensor inputs (aligned with OpenCV version)

    Args:
        pred_tensor: Predicted image tensor [B,C,H,W] or [C,H,W]
        target_tensor: Target image tensor (must match shape)
        data_range: Pixel value range (255 or 1)
        average: Return average (True) or sum (False)

    Returns:
        Tuple of (PSNR result, SSIM result)
    """
    # Convert to NumPy and ensure CPU
    pred_np = pred_tensor.detach().cpu().numpy()
    target_np = target_tensor.detach().cpu().numpy()

    # Initialize results storage
    psnr_values = []
    ssim_values = []

    # Handle both single and batch inputs
    if len(pred_np.shape) == 3:  # Single image [C,H,W]
        pred_np = [np.transpose(pred_np, (1, 2, 0))]  # Convert to [H,W,C] in list
        target_np = [np.transpose(target_np, (1, 2, 0))]
    else:  # Batch [B,C,H,W]
        pred_np = [np.transpose(p, (1, 2, 0)) for p in pred_np]  # Convert each to [H,W,C]
        target_np = [np.transpose(t, (1, 2, 0)) for t in target_np]

    # Process each image
    for p, t in zip(pred_np, target_np):
        # Convert to float64 (consistent with OpenCV version)
        p = p.astype(np.float64)
        t = t.astype(np.float64)

        # Convert to grayscale using OpenCV's method (consistent with OpenCV version)
        if p.shape[2] == 3:  # RGB image
            # Convert to uint8 (OpenCV expects 0-255 for grayscale conversion)
            p_uint8 = (p * 255.0).clip(0, 255).astype(np.uint8)
            t_uint8 = (t * 255.0).clip(0, 255).astype(np.uint8)

            # Convert to grayscale using OpenCV (consistent with OpenCV version)
            p_gray = cv2.cvtColor(p_uint8, cv2.COLOR_RGB2GRAY).astype(np.float64) / 255.0 * data_range
            t_gray = cv2.cvtColor(t_uint8, cv2.COLOR_RGB2GRAY).astype(np.float64) / 255.0 * data_range
        else:  # Already grayscale
            p_gray = p.squeeze()
            t_gray = t.squeeze()

        # Calculate PSNR
        psnr = peak_signal_noise_ratio(
            t_gray, p_gray,  # Note: OpenCV uses (ref, test) order
            data_range=data_range
        )
        psnr_values.append(psnr)

        # Calculate SSIM (use same parameters as OpenCV version)
        ssim = structural_similarity(
            t_gray, p_gray,  # Note: OpenCV uses (ref, test) order
            data_range=data_range
        )
        ssim_values.append(ssim)

    # Return results
    return (np.mean(psnr_values), np.mean(ssim_values)) if average else (sum(psnr_values), sum(ssim_values))


def calculate_batch_psnr_ssim_color(real_dir, fake_dir, data_range=255.0, target_size=(128, 128)):
    """
    Calculate PSNR and SSIM for color images (no grayscale conversion)

    Args:
        real_dir: Directory containing reference images
        fake_dir: Directory containing generated images
        data_range: Pixel value range (255 for [0,255], 1 for [0,1])
        target_size: Target image size (width, height)

    Returns:
        Tuple of (mean PSNR, mean SSIM)
    """
    # Get matched file lists
    real_files = sorted([f for f in os.listdir(real_dir)
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    fake_files = sorted([f for f in os.listdir(fake_dir)
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    assert len(real_files) == len(fake_files), "Mismatched number of files"

    assert real_files == fake_files, "Filename mismatch"

    psnr_values = []
    ssim_values = []
    error_files = []

    for filename in tqdm(real_files, desc="Calculate Color PSNR & SSIM"):
        try:
            # Load images (BGR format)
            real_img = cv2.imread(os.path.join(real_dir, filename), cv2.IMREAD_COLOR)
            fake_img = cv2.imread(os.path.join(fake_dir, filename), cv2.IMREAD_COLOR)

            # Validate loading
            if real_img is None or fake_img is None:
                raise ValueError("Image loading failed")

            # Resize to target dimensions
            real_img = cv2.resize(real_img, target_size, interpolation=cv2.INTER_AREA)
            fake_img = cv2.resize(fake_img, target_size, interpolation=cv2.INTER_AREA)

            # Convert BGR to RGB
            real_img = cv2.cvtColor(real_img, cv2.COLOR_BGR2RGB)
            fake_img = cv2.cvtColor(fake_img, cv2.COLOR_BGR2RGB)

            # Convert to float and normalize
            real_img = real_img.astype(np.float64)
            fake_img = fake_img.astype(np.float64)
            if data_range == 1.0:
                real_img /= 255.0
                fake_img /= 255.0

            # Calculate color PSNR (use all three channels)
            psnr = peak_signal_noise_ratio(real_img, fake_img, data_range=data_range)

            # Calculate color SSIM (multi-channel mode)
            ssim = structural_similarity(
                real_img, fake_img,
                data_range=data_range,
                channel_axis=2  # Specify channel dimension
            )

            psnr_values.append(psnr)
            ssim_values.append(ssim)

        except Exception as e:
            error_files.append(filename)
            print(f"\nProcessing failed: {filename} - Error: {str(e)}")
            continue

    if error_files:
        print(f"\nWarning: {len(error_files)} files failed processing")
        with open("error_log_color.txt", "w") as f:
            f.write("\n".join(error_files))

    return np.mean(psnr_values), np.mean(ssim_values)


def calculate_batch_psnr_ssim_pt_color(pred_tensor, target_tensor, data_range=1.0, average=False):
    """
    Calculate PSNR and SSIM for color tensor inputs (no grayscale conversion)

    Args:
        pred_tensor: Predicted image tensor [B,C,H,W] or [C,H,W]
        target_tensor: Target image tensor (must match shape)
        data_range: Pixel value range (255 or 1)
        average: Return average (True) or sum (False)

    Returns:
        Tuple of (PSNR result, SSIM result)
    """
    # Convert to NumPy and ensure CPU
    pred_np = pred_tensor.detach().cpu().numpy()
    target_np = target_tensor.detach().cpu().numpy()

    # Initialize results storage
    psnr_values = []
    ssim_values = []

    # Handle single and batch inputs
    if len(pred_np.shape) == 3:  # Single image [C,H,W]
        pred_np = [np.transpose(pred_np, (1, 2, 0))]  # Convert to [H,W,C]
        target_np = [np.transpose(target_np, (1, 2, 0))]
    else:  # Batch [B,C,H,W]
        pred_np = [np.transpose(p, (1, 2, 0)) for p in pred_np]  # Convert each to [H,W,C]
        target_np = [np.transpose(t, (1, 2, 0)) for t in target_np]

    # Process each image
    for p, t in zip(pred_np, target_np):
        # Convert to float64
        p = p.astype(np.float64)
        t = t.astype(np.float64)

        # Normalize if necessary
        if data_range == 1.0 and p.max() > 1.0:
            p /= 255.0
            t /= 255.0

        # Calculate color PSNR
        psnr = peak_signal_noise_ratio(
            t, p,  # Order: (reference, test)
            data_range=data_range
        )
        psnr_values.append(psnr)

        # Calculate color SSIM (multi-channel mode)
        ssim = structural_similarity(
            t, p,  # Order: (reference, test)
            data_range=data_range,
            channel_axis=2  # Specify channel dimension
        )
        ssim_values.append(ssim)

    # Return results
    return (np.mean(psnr_values), np.mean(ssim_values)) if average else (sum(psnr_values), sum(ssim_values))


if __name__ == '__main__':
    # Set target size and batch size
    TARGET_SIZE = (256, 256)
    BATCH_SIZE = 1

    # Path configuration
    target_path = ''
    fake_path = ''

    # Calculate metrics
    psnr, ssim = calculate_batch_psnr_ssim_color(target_path, fake_path, target_size=TARGET_SIZE)
    lpips_value = calculate_batch_lpips(target_path, fake_path, batch_size=BATCH_SIZE)

    print('======Calculate FID======')
    # Calculate FID
    fid_value = calculate_fid_given_paths([target_path, fake_path], batch_size=BATCH_SIZE, device='cuda', dims=2048)

    print("[PSNR: %.4f\t SSIM: %.4f\t LPIPS: %.4f\t FID: %.4f]" % (psnr, ssim, lpips_value, fid_value))