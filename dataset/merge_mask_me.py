import os
import torch
from PIL import Image
import torchvision.transforms as transforms
from torchvision.utils import save_image
from tqdm import tqdm

# 设置路径
image_dir = "/mnt/disk1/ruanwentao/data/ARMCD/train/target/"
mask_dir = "/mnt/disk1/ruanwentao/data/ARMCD/train/mask"
output_dir = "/mnt/disk1/ruanwentao/data/ARMCD/train/input"

# 创建输出目录
os.makedirs(output_dir, exist_ok=True)

# 定义与原代码相同的转换
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# 获取所有图像文件
image_files = sorted([f for f in os.listdir(image_dir) if f.startswith("train_") and f.endswith(".jpg")])

# 处理每对图像和掩码
for image_file in tqdm(image_files, desc="Processing images"):
    # 提取序号
    index = image_file.split("_")[1].split(".")[0]
    mask_file = f"mask_{index}.jpg"

    # 构建完整路径
    image_path = os.path.join(image_dir, image_file)
    mask_path = os.path.join(mask_dir, mask_file)
    output_path = os.path.join(output_dir, f"train_{index}.jpg")

    # 检查文件是否存在
    if not os.path.exists(mask_path):
        print(f"警告: 找不到对应掩码 {mask_path}，跳过")
        continue

    # 打开并转换图像和掩码（完全遵循原代码逻辑）
    img = Image.open(image_path).convert('RGB')
    mask = Image.open(mask_path).convert('RGB')

    # 应用相同的变换
    img = transform(img)
    mask = transform(mask)

    # 逐元素相乘（完全遵循原代码逻辑）
    result = torch.mul(img, mask)

    # 保存结果
    save_image(result, output_path)

print("所有图像处理完成！")