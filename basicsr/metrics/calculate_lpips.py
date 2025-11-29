import numpy as np
import torch
import lpips

from torchvision import transforms
from PIL import Image
import os

from tqdm import tqdm


def calculate_batch_lpips(real_dir, fake_dir, net='vgg', device='cuda', batch_size=16):
    """
    优化后的LPIPS计算函数，支持批处理

    参数:
        real_dir (str): 真实图像目录
        fake_dir (str): 生成图像目录
        net (str): 模型类型('alex'|'vgg'|'squeeze')
        device (str): 计算设备
        batch_size (int): 批处理大小

    返回:
        float: 平均LPIPS值
    """
    # 初始化模型
    loss_fn = lpips.LPIPS(net=net).to(device).eval()
    for param in loss_fn.parameters():
        param.requires_grad = False

    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize(256),  # 调整为推荐尺寸
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # 加载图像路径并验证
    real_paths = sorted([os.path.join(real_dir, f) for f in os.listdir(real_dir)
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    fake_paths = sorted([os.path.join(fake_dir, f) for f in os.listdir(fake_dir)
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    assert len(real_paths) == len(fake_paths), "文件数量不匹配"
    assert all(os.path.basename(r) == os.path.basename(f)
               for r, f in zip(real_paths, fake_paths)), "文件名不匹配"

    # 批处理计算
    lpips_values = []
    for i in tqdm(range(0, len(real_paths), batch_size),
                  desc=f"LPIPS ({net})", unit="batch"):
        # 加载批次
        real_batch = torch.stack([
            transform(Image.open(p).convert('RGB'))
            for p in real_paths[i:i + batch_size]
        ]).to(device)

        fake_batch = torch.stack([
            transform(Image.open(p).convert('RGB'))
            for p in fake_paths[i:i + batch_size]
        ]).to(device)

        # 计算LPIPS
        with torch.no_grad():
            batch_lpips = loss_fn(real_batch, fake_batch)
            lpips_values.extend(batch_lpips.cpu().numpy())

    return np.mean(lpips_values)

def calculate_single_lpips(real_image_path, fake_image_path, net='vgg', device='cuda'):
    """
    计算单张图像的 LPIPS（Learned Perceptual Image Patch Similarity）。

    参数:
        real_image_path (str): 真实图像的路径。
        fake_image_path (str): 生成图像的路径。
        net (str): 使用的预训练模型，可选 'alex', 'vgg', 'squeeze'，默认为 'vgg'。
        device (str): 计算设备，默认为 'cuda'。

    返回:
        float: LPIPS 值。
    """
    # 初始化 LPIPS 模型
    loss_fn = lpips.LPIPS(net=net).to(device)

    # 图像预处理函数
    def load_image(image_path):
        image = Image.open(image_path).convert('RGB')
        transform = transforms.Compose([
            transforms.ToTensor(),  # 转换为 Tensor
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 归一化到 [-1, 1]
        ])
        return transform(image).unsqueeze(0).to(device)  # 添加 batch 维度并移动到指定设备

    # 加载图像
    real_image = load_image(real_image_path)
    fake_image = load_image(fake_image_path)

    # 计算 LPIPS
    lpips_value = loss_fn(real_image, fake_image)
    return lpips_value.item()



if __name__ == '__main__':
    # 定义真实图像和生成图像的目录
    real_dir = '/mnt/disk1/ruanwentao/code/ImageRestoration/test/Restormer_result/real_images/'
    fake_dir = '/mnt/disk1/ruanwentao/code/ImageRestoration/test/Restormer_result/fake_images/'

    # 确保目录存在
    if not os.path.exists(real_dir) or not os.path.exists(fake_dir):
        raise FileNotFoundError("真实图像或生成图像的目录不存在！")

    # 计算 LPIPS
    mean_lpips = calculate_batch_lpips(real_dir, fake_dir, net='vgg', device='cuda')
    print(f"Mean LPIPS: {mean_lpips:.4f}")

    print("*"*20)
    real_images = sorted(os.listdir(real_dir))
    fake_images = sorted(os.listdir(fake_dir))
    lpips_values = []
    for real_name, fake_name in zip(real_images, fake_images):
        real_image_path = os.path.join(real_dir, real_name)
        fake_image_path = os.path.join(fake_dir, fake_name)

        # 计算 LPIPS
        lpips_value = calculate_single_lpips(real_image_path, fake_image_path, net='vgg', device='cuda')
        lpips_values.append(lpips_value)

    # 计算平均 LPIPS
    mean1_lpips = sum(lpips_values) / len(lpips_values)

    print(f"Mean1 LPIPS: {mean1_lpips:.4f}")
