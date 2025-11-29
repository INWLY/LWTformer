import os
from PIL import Image


def convert_to_png(input_dir):
    """将指定目录中的所有图片转换为PNG格式"""
    # 确保输入目录存在
    if not os.path.exists(input_dir):
        print(f"错误：目录 {input_dir} 不存在")
        return

    # 支持的图片格式
    supported_formats = ('.jpg', '.jpeg', '.bmp', '.webp')

    # 遍历目录中的所有文件
    for filename in os.listdir(input_dir):
        filepath = os.path.join(input_dir, filename)

        # 跳过子目录
        if os.path.isdir(filepath):
            continue

        # 检查文件是否为支持的图片格式
        if filename.lower().endswith(supported_formats):
            try:
                # 打开图片
                with Image.open(filepath) as img:
                    # 构建新的文件名（替换扩展名或直接添加.png）
                    base_name = os.path.splitext(filename)[0]
                    new_filename = f"{base_name}.png"
                    new_filepath = os.path.join(input_dir, new_filename)

                    # 转换并保存为PNG
                    img.save(new_filepath, 'PNG')
                    print(f"已转换: {filename} -> {new_filename}")

                    # 可选：删除原始文件
                    os.remove(filepath)

            except Exception as e:
                print(f"处理文件 {filename} 时出错: {e}")
        else:
            print(f"跳过非图片文件: {filename}")


if __name__ == "__main__":
    input_directory = "/mnt/disk1/ruanwentao/data/ARMCD/train/input"
    convert_to_png(input_directory)
    print("图片转换完成！")