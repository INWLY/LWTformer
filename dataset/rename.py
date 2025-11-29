import os
import re

directory = '/mnt/disk1/ruanwentao/data/ARMCD/train/target'

for filename in os.listdir(directory):
    # 使用正则表达式匹配 eroded_数字.jpg 格式
    match = re.match(r'^real_(\d+)\.jpg$', filename)
    if match:
        number = match.group(1)  # 获取数字部分
        new_filename = f'train_{number}.jpg'
        os.rename(
            os.path.join(directory, filename),
            os.path.join(directory, new_filename)
        )
        print(f"已重命名: {filename} -> {new_filename}")