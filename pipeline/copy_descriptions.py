import os
import shutil
import re
from pathlib import Path


def main():
    # 定义路径（相对于项目根路径）
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from config.paths import DATA_ROOT, random_dataset

    meme_dir = random_dataset("meme")
    test_data_dir = DATA_ROOT / "test_data_discription_2.0"
    train_data_dir = DATA_ROOT / "train_data_discription_2.0"
    target_dir = random_dataset("raw_message")

    # 确保目标目录存在
    target_dir.mkdir(parents=True, exist_ok=True)

    # 检查源目录是否存在
    if not meme_dir.exists():
        print(f"错误：图片目录不存在 {meme_dir}")
        return

    if not test_data_dir.exists() and not train_data_dir.exists():
        print(f"错误：描述数据目录不存在")
        print(f"检查路径：{test_data_dir}")
        print(f"检查路径：{train_data_dir}")
        return

    # 获取所有图片文件
    image_files = []
    for file in meme_dir.glob("*.jpg"):
        # 检查文件名是否符合"数字.jpg"格式
        if re.match(r'^\d+\.jpg$', file.name):
            image_files.append(file)

    if not image_files:
        print(f"在 {meme_dir} 中没有找到符合格式的图片文件")
        return

    print(f"找到 {len(image_files)} 个图片文件")

    copied_count = 0
    not_found_count = 0

    for img_file in sorted(image_files):
        # 构造对应的JSON文件名
        json_filename = f"{img_file.name}.json"

        # 在两个可能的目录中查找JSON文件
        json_file_path = None

        if test_data_dir.exists():
            test_json_path = test_data_dir / json_filename
            if test_json_path.exists():
                json_file_path = test_json_path

        if json_file_path is None and train_data_dir.exists():
            train_json_path = train_data_dir / json_filename
            if train_json_path.exists():
                json_file_path = train_json_path

        if json_file_path:
            # 复制JSON文件到目标目录
            target_file = target_dir / json_filename
            try:
                shutil.copy2(json_file_path, target_file)
                print(f"✓ 已复制: {json_filename}")
                copied_count += 1
            except Exception as e:
                print(f"✗ 复制失败 {json_filename}: {e}")
        else:
            print(f"✗ 未找到对应的JSON文件: {json_filename}")
            not_found_count += 1

    # 输出统计信息
    print(f"\n=== 复制完成 ===")
    print(f"总图片文件数: {len(image_files)}")
    print(f"成功复制: {copied_count}")
    print(f"未找到JSON: {not_found_count}")
    print(f"目标目录: {target_dir}")


if __name__ == "__main__":
    main()