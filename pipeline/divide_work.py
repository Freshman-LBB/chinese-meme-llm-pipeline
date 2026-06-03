import os
from pathlib import Path
import re

def natural_sort_key(s):
    """
    用于自然排序的关键函数，确保数字按照正确的顺序排序
    """
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', s)]

def main():
    # 获取项目根目录
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from config.paths import REPO_ROOT, random_dataset

    project_root = REPO_ROOT

    # 目标目录
    meme_dir = random_dataset("meme")

    # 检查目录是否存在
    if not meme_dir.exists():
        print(f"目录不存在: {meme_dir}")
        return

    # 获取所有符合 数字.jpg.json 格式的文件
    json_files = [f for f in meme_dir.glob('*.jpg')]

    # 如果没有文件
    if not json_files:
        print("目录中没有找到任何jpg文件")
        return

    # 按文件名中的数字自然排序
    sorted_files = sorted(json_files, key=lambda x: natural_sort_key(x.stem))

    # 打印总文件数
    total_files = len(sorted_files)
    print(f"总共找到 {total_files} 个文件")

    # 询问分组大小
    while True:
        try:
            group_size = int(input("请输入每组的文件数量: "))
            if group_size <= 0:
                print("请输入一个正整数")
                continue
            break
        except ValueError:
            print("请输入有效的数字")

    # 计算分组
    groups = []
    for i in range(0, total_files, group_size):
        start_num = int(sorted_files[i].stem.split('.')[0])
        end_index = min(i + group_size - 1, total_files - 1)
        end_num = int(sorted_files[end_index].stem.split('.')[0])
        groups.append(f"[{start_num}, {end_num}] {group_size}")

    # 输出路径
    output_file = REPO_ROOT / "pipeline" / "divide_work.txt"

    # 写入文件
    with open(output_file, 'w', encoding='utf-8') as f:
        for group in groups:
            f.write(group + '\n')

    print(f"分组结果已保存到 {output_file}")
    print("\n分组详情：")
    for group in groups:
        print(group)

if __name__ == "__main__":
    main()