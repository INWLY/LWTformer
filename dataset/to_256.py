import os
from PIL import Image
import torchvision.transforms as transforms


def resize_images(input_dir, target_size=(256, 256)):
    """
    将指定目录下的所有图片调整为目标尺寸
    :param input_dir: 图片所在目录路径
    :param target_size: 目标尺寸，默认(256, 256)
    """
    # 检查目录是否存在
    if not os.path.exists(input_dir):
        print(f"错误：目录 {input_dir} 不存在")
        return

    # 定义 resize 转换（使用高质量插值）
    resize_transform = transforms.Compose([
        transforms.Resize(target_size, interpolation=Image.LANCZOS),
    ])

    # 遍历目录中的所有文件
    for filename in os.listdir(input_dir):
        file_path = os.path.join(input_dir, filename)

        # 跳过目录，只处理文件
        if os.path.isdir(file_path):
            continue

        # 尝试打开图片文件（过滤非图片文件）
        try:
            with Image.open(file_path) as img:
                # 检查当前尺寸是否已为目标尺寸
                if img.size == target_size:
                    print(f"已跳过（尺寸匹配）：{filename}")
                    continue

                # 调整尺寸
                resized_img = resize_transform(img)

                # 保存（覆盖原文件，保持原格式）
                resized_img.save(file_path)
                print(f"已调整：{filename} （原尺寸：{img.size} → 新尺寸：{target_size}）")

        except Exception as e:
            print(f"处理失败 {filename}：{str(e)}")


if __name__ == "__main__":
    # 目标路径
    input_directory = "/mnt/disk1/ruanwentao/data/ARMCD/test/input"
    # 调用函数，将图片调整为256×256
    resize_images(input_directory, target_size=(256, 256))
    print("所有图片处理完成！")