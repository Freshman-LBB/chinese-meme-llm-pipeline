import os
import json
import csv
from collections import defaultdict, Counter


def merge_labels(files):
    merged_data = None
    for file in files:
        try:
            with open(file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            print("\n=== JSON 解析失败 ===")
            print("文件路径:", file)
            print("错误信息:", e)
            # 如果你不想继续处理后面的文件，可以直接抛出异常停止程序
            raise

        if merged_data is None:
            # 确保是 dict
            if not isinstance(data, dict):
                print("\n警告: 文件不是 JSON 对象(dict):", file)
                print("实际类型:", type(data))
                raise TypeError("期望 JSON 对象(dict)")
            merged_data = data.copy()

        for label in ['label_1', 'label_2', 'label_3', 'label_4', 'label_5']:
            if label in data:
                merged_data[label] = data[label]

    return merged_data


def check_and_set_label(merged_data):
    # 检查 label_1 到 label_5 的数量和一致性
    labels = ['label_1', 'label_2', 'label_3', 'label_4', 'label_5']
    existing_labels = [label for label in labels if label in merged_data]

    # 获取所有存在的标签的具体数值
    label_values = [merged_data[label] for label in existing_labels]

    # 修改点1: 数量不是3个才把lable设置为4
    if len(existing_labels) != 3:
        merged_data['label'] = 4
    else:
        # 统计这3个值的出现频率
        # 例如: [0, 0, 1] -> Counter({0: 2, 1: 1})
        # 例如: [0, 1, 2] -> Counter({0: 1, 1: 1, 2: 1})
        counts = Counter(label_values)

        # 获取出现次数最多的元素及其次数
        most_common = counts.most_common()  # 返回列表，如 [(0, 2), (1, 1)] 或 [(0, 1), (1, 1), (2, 1)]

        top_value, top_count = most_common[0]

        # 修改点2: 如果三个标签各不同值就设为3 (此时最高频次为1)
        if top_count == 1:
            merged_data['label'] = 3
        else:
            # 遵循多数原则，值设为0，1，2 (即设为出现次数最多的那个值)
            merged_data['label'] = top_value

    return merged_data


def process_dataset(base_path):
    # 原有的目录创建逻辑
    os.makedirs(os.path.join(base_path, 'construct', 'dataset', 'harmful'), exist_ok=True)
    os.makedirs(os.path.join(base_path, 'construct', 'dataset', 'harmless'), exist_ok=True)

    # 用于统计的数据结构
    label_stats = defaultdict(lambda: {
        'total': 0,
        'harmful_num': 0,
        'harmless_num': 0,
        'harmful_files': [],
        'harmless_files': []
    })

    # 处理harmful和harmless数据集
    for dataset_type in ['harmful', 'harmless']:
        file_groups = {}
        # 这里的文件夹列表如果你有变化请自行调整
        for folder in ['li', 'lu', 'cao', 'sheng', 'yao']:
            folder_path = os.path.join(base_path, 'random', folder, dataset_type)

            if not os.path.exists(folder_path):
                print(f"Warning: Path not found {folder_path}")
                continue

            for filename in os.listdir(folder_path):
                if filename.endswith('.json'):
                    if filename not in file_groups:
                        file_groups[filename] = []
                    file_groups[filename].append(os.path.join(folder_path, filename))

        # 合并每组文件
        for filename, files in file_groups.items():
            merged_data = merge_labels(files)
            merged_data = check_and_set_label(merged_data)

            # 更新统计信息
            label = merged_data.get('label', 'unknown')
            label_stats[label]['total'] += 1
            label_stats[label][f'{dataset_type}_num'] += 1
            label_stats[label][f'{dataset_type}_files'].append(filename)

            # 输出到目标文件夹
            output_path = os.path.join(base_path, 'construct', 'dataset', dataset_type, filename)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(merged_data, f, ensure_ascii=False, indent=2)

    # 输出CSV统计信息
    csv_path = os.path.join(base_path, 'construct', 'dataset', 'label_stats.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['label', 'all-num', 'harmful-num', 'harmless-num', 'harmful-files', 'harmless-files']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for label, stats in label_stats.items():
            writer.writerow({
                'label': label,
                'all-num': stats['total'],
                'harmful-num': stats['harmful_num'],
                'harmless-num': stats['harmless_num'],
                'harmful-files': ','.join(stats['harmful_files']),
                'harmless-files': ','.join(stats['harmless_files'])
            })

    print("数据集合并完成，统计信息已输出到 label_stats.csv")


def main():
    # 获取项目根目录
    # 注意：这里假设脚本位置没有变，如果路径不对请手动指定 base_path
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from config.paths import DATA_ROOT
    project_root = str(DATA_ROOT)
    # 也可以直接写死路径测试，例如:
    # project_root = r"D:\YourProjectData"
    process_dataset(project_root)


if __name__ == '__main__':
    main()