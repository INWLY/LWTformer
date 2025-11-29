import torch
import numpy as np
import pickle
import cv2
from skimage.metrics import structural_similarity

import utils


def is_numpy_file(filename):
    return any(filename.endswith(extension) for extension in [".npy"])

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".jpg"])

def is_png_file(filename):
    return any(filename.endswith(extension) for extension in [".png"])

def is_pkl_file(filename):
    return any(filename.endswith(extension) for extension in [".pkl"])

def load_pkl(filename_):
    with open(filename_, 'rb') as f:
        ret_dict = pickle.load(f)
    return ret_dict    

def save_dict(dict_, filename_):
    with open(filename_, 'wb') as f:
        pickle.dump(dict_, f)    

def load_npy(filepath):
    img = np.load(filepath)
    return img

def load_img(filepath):
    img = cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)
    img = img/255.
    return img

def save_img(filepath, img):
    if isinstance(img, torch.Tensor):
        # 将 torch.Tensor 转换为 numpy 数组
        img = img.cpu().detach().numpy()
        # 如果 img 是 4 维张量（batch, channels, height, width），假设只处理第一张图片
        if len(img.shape) == 4:
            img = img[0]
        # 调整通道顺序，从 (C, H, W) 到 (H, W, C)
        img = np.transpose(img, (1, 2, 0))

        # 确保像素值在 [0, 255] 范围内
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    cv2.imwrite(filepath,cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

def myPSNR(tar_img, prd_img):
    imdff = torch.clamp(prd_img,0,1) - torch.clamp(tar_img,0,1)
    rmse = (imdff**2).mean().sqrt()
    ps = 20*torch.log10(1/rmse)
    return ps

def batch_PSNR(img1, img2, average=False):
    PSNR = []
    for im1, im2 in zip(img1, img2):
        psnr = myPSNR(im1, im2)
        PSNR.append(psnr)
    return sum(PSNR)/len(PSNR) if average else sum(PSNR)

# --------------------------------------------
# SSIM
# --------------------------------------------
def ssim(img1, img2):
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()



def batch_SSIM(img1_batch, img2_batch, average=False):
    """
    计算批量图像的 SSIM 值
    :param img1_batch: 批量图像 1，形状为 (batch_size, channels, height, width)
    :param img2_batch: 批量图像 2，形状为 (batch_size, channels, height, width)
    :return: 每张图像的 SSIM 值列表
    """
    # 检查输入的两个批量图像的形状是否一致
    if img1_batch.shape != img2_batch.shape:
        raise ValueError("Input image batches must have the same shape.")

    # 获取批量大小
    batch_size = img1_batch.shape[0]
    ssim_values = []

    # 遍历批量中的每张图像
    for i in range(batch_size):
        img1 = img1_batch[i]
        img2 = img2_batch[i]

        # 将张量从 GPU 复制到 CPU 并转换为 NumPy 数组
        if isinstance(img1, torch.Tensor):
            img1 = img1.cpu().numpy()
        if isinstance(img2, torch.Tensor):
            img2 = img2.cpu().numpy()

        # 将通道维度移到最后，符合 structural_similarity 函数的要求
        img1 = np.transpose(img1, (1, 2, 0))
        img2 = np.transpose(img2, (1, 2, 0))

        # 判断图像像素值范围，确定 data_range
        if img1.dtype in [np.float32, np.float64]:
            if img1.max() <= 1.0:
                data_range = 1.0
            else:
                data_range = 255.0
        else:
            data_range = 255.0

        if len(img1.shape) == 2:  # 灰度图像
            SSIM = structural_similarity(img1, img2, win_size=3, data_range=data_range)
        elif len(img1.shape) == 3:  # RGB 图像
            SSIM = structural_similarity(img1, img2, win_size=3, channel_axis=2, data_range=data_range)
        else:
            raise ValueError("Unsupported image shape: {}".format(img1.shape))

        ssim_values.append(SSIM)

    return sum(ssim_values)/len(ssim_values) if average else sum(ssim_values)
