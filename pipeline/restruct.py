import json
import re
from pathlib import Path


def load_json_file(file_path):
    """安全加载JSON文件"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"读取JSON文件失败 {file_path}: {e}")
        return None


def save_json_file(file_path, data):
    """安全保存JSON文件"""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        print(f"保存JSON文件失败 {file_path}: {e}")
        return False


def extract_number_from_path(path_value):
    """从path字段中提取数字"""
    if not path_value:
        return None

    # 移除.jpg后缀
    name_without_ext = path_value.replace('.jpg', '')

    # 如果是纯数字，直接返回
    if name_without_ext.isdigit():
        return name_without_ext

    # 尝试提取开头的数字
    match = re.match(r'^(\d+)', name_without_ext)
    if match:
        return match.group(1)

    return None


def process_single_json(json_file, output_dir, is_harmful):
    """处理单个JSON文件"""
    data = load_json_file(json_file)
    if data is None:
        return 0

    # 检查必要字段
    required_fields = ['path', 'raw_label', 'text', 'scene', 'label']
    missing_fields = [field for field in required_fields if field not in data]

    if missing_fields:
        print(f"警告: {json_file.name} 缺少字段: {missing_fields}")
        return 0

    # 检查label值（保留原逻辑：只接受 0/1）
    label = data.get('label')
    if label not in [0, 1]:
        print(f"跳过 {json_file.name}: label值为 {label}")
        return 0

    # 提取数据
    path_value = data.get('path', '')
    number = extract_number_from_path(path_value)

    if not number:
        print(f"警告: 无法从 {json_file.name} 的path字段提取数字: {path_value}")
        return 0

    # 准备输出数据（保持不变）
    output_data = {
        "path": path_value,
        "raw_label": data.get('raw_label'),
        "text": data.get('text', ''),
        "scene": data.get('scene', ''),
        "label": label
    }

    # ★★★ 关键修改：后缀由“目录类型”决定，而不是 label ★★★
    filename_suffix = "-P.json" if is_harmful else "-N.json"
    output_filename = f"{number}{filename_suffix}"
    output_file_path = output_dir / output_filename

    # 保存文件
    if save_json_file(output_file_path, output_data):
        print(f"  ✓ 创建文件: {output_filename}")
        return 1
    else:
        print(f"  ✗ 创建文件失败: {output_filename}")
        return 0


def process_directory(source_dir, output_dir, is_harmful):
    """处理目录中的所有JSON文件"""
    if not source_dir.exists():
        print(f"警告：目录不存在 {source_dir}")
        return 0, 0

    json_files = list(source_dir.glob("*.json"))

    if not json_files:
        print(f"在 {source_dir} 中没有找到JSON文件")
        return 0, 0

    print(f"\n=== 处理 {source_dir.name} 目录 ===")
    print(f"路径: {source_dir}")
    print(f"找到 {len(json_files)} 个JSON文件")

    total_created = 0
    total_processed = 0

    for json_file in json_files:
        print(f"处理: {json_file.name}")
        created = process_single_json(json_file, output_dir, is_harmful)
        total_created += created
        total_processed += 1

    print(f"{source_dir.name} 目录处理完成:")
    print(f"  处理文件: {total_processed}")
    print(f"  创建文件: {total_created}")

    return total_created, total_processed


def main():
    # 定义路径
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from config.paths import construct_dataset

    harmful_dir = construct_dataset("harmful")
    harmless_dir = construct_dataset("harmless")
    output_dir = construct_dataset("restructured")

    print("=== JSON文件重构工具 ===")
    print("重构有害和无害数据集")

    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n输出目录: {output_dir}")

    # 清空输出目录中的现有JSON文件
    existing_json_files = list(output_dir.glob("*.json"))
    if existing_json_files:
        print(f"清理输出目录中的 {len(existing_json_files)} 个现有JSON文件...")
        for existing_file in existing_json_files:
            try:
                existing_file.unlink()
                print(f"  删除: {existing_file.name}")
            except Exception as e:
                print(f"  删除文件失败 {existing_file.name}: {e}")

    total_created_files = 0
    total_processed_files = 0

    # 处理 harmful 目录
    created, processed = process_directory(harmful_dir, output_dir, is_harmful=True)
    total_created_files += created
    total_processed_files += processed

    # 处理 harmless 目录：统一输出为 -N
    created, processed = process_directory(harmless_dir, output_dir, is_harmful=False)
    total_created_files += created
    total_processed_files += processed

    # 统计输出目录中的文件
    all_output_files = list(output_dir.glob("*.json"))
    p_files = [f for f in all_output_files if "-P.json" in f.name]
    n_files = [f for f in all_output_files if "-N.json" in f.name]

    print(f"\n=== 处理完成 ===")
    print(f"总处理文件: {total_processed_files}")
    print(f"总创建文件: {total_created_files}")

    print(f"\n=== 输出文件统计 ===")
    print(f"有害文件 (-P): {len(p_files)}")
    print(f"无害文件 (-N): {len(n_files)}")
    print(f"总输出文件: {len(all_output_files)}")

    print(f"\n所有输出文件已统一保存到: {output_dir}")

    # 显示一些示例文件名
    if all_output_files:
        print(f"\n示例文件名:")
        sorted_files = sorted(all_output_files, key=lambda x: x.name)
        for i, file in enumerate(sorted_files[:15]):
            # 显示文件的基本信息
            data = load_json_file(file)
            if data:
                label = data.get('label', '?')
                print(f"  {file.name} (label={label})")
            else:
                print(f"  {file.name}")

        if len(all_output_files) > 15:
            print(f"  ... 还有 {len(all_output_files) - 15} 个文件")


if __name__ == "__main__":
    main()