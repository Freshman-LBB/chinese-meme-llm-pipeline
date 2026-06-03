import os
import json
import csv
import numpy as np
from sklearn.metrics import cohen_kappa_score


def load_json_files(base_path):
    """
    加载harmless和harmful文件夹中label为0或1的JSON文件
    """
    harmless_path = os.path.join(base_path, 'construct', 'dataset', 'harmless')
    harmful_path = os.path.join(base_path, 'construct', 'dataset', 'harmful')

    # 存储文件名与标签的映射
    first_annotator_labels = {}
    second_annotator_labels = {}

    # 处理harmless文件夹（标记为0）
    for filename in os.listdir(harmless_path):
        if filename.endswith('.json'):
            filepath = os.path.join(harmless_path, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if data.get('label') in [0, 1]:
                    first_annotator_labels[filepath] = data.get('label')
                    second_annotator_labels[filepath] = 0

    # 处理harmful文件夹（标记为1）
    for filename in os.listdir(harmful_path):
        if filename.endswith('.json'):
            filepath = os.path.join(harmful_path, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if data.get('label') in [0, 1]:
                    first_annotator_labels[filepath] = data.get('label')
                    second_annotator_labels[filepath] = 1

    print("Total files loaded:", len(first_annotator_labels))
    return first_annotator_labels, second_annotator_labels


def read_divide_work(filepath):
    """
    读取分组信息
    """
    groups = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            match = line.split(']')
            range_part = match[0][1:].split(',')
            start = int(range_part[0])
            end = int(range_part[1])
            groups.append((start, end))

    return groups


def calculate_kappa(first_labels, second_labels, file_subset=None):
    """
    计算总体Kappa值
    - first_labels, second_labels: dict[filepath] -> label
    - file_subset: 可选，限定参与计算的一批文件路径（严格模式会用到）
    """
    if file_subset is None:
        # 不严格模式：所有文件
        keys = list(first_labels.keys())
    else:
        # 严格模式：只在给定子集里
        keys = list(file_subset)

    if len(keys) == 0:
        return None, []

    labels1 = [first_labels[k] for k in keys]
    labels2 = [second_labels[k] for k in keys]

    total_kappa = cohen_kappa_score(labels1, labels2)
    return total_kappa, keys


def calculate_group_kappa(first_labels, second_labels, groups, file_subset=None):
    """
    计算分组Kappa值
    - groups: [(start, end), ...]
    - file_subset: 可选，限定参与计算的一批文件路径（严格模式会用到）
    """
    # 如果传入子集，则先转成 set，加速判断
    allowed_files = set(file_subset) if file_subset is not None else None

    group_results = []

    for group in groups:
        group_first_labels = {}
        group_second_labels = {}

        # 筛选在分组范围内的文件
        for filepath, label in first_labels.items():
            # 如果有限定子集，但当前文件不在其中，则跳过（严格模式）
            if allowed_files is not None and filepath not in allowed_files:
                continue

            filename = os.path.basename(filepath)
            # 假设文件名格式为数字.jpg.json
            try:
                file_num = int(filename.split('.')[0])
            except ValueError:
                # 如果文件名不是数字开头，直接跳过
                continue

            if group[0] <= file_num <= group[1]:
                group_first_labels[filepath] = label
                group_second_labels[filepath] = second_labels[filepath]

        # 按文件名中的数字排序
        sorted_group_files = sorted(
            group_first_labels.keys(),
            key=lambda x: int(os.path.basename(x).split('.')[0])
        )

        if len(sorted_group_files) < 2:
            group_results.append({
                'group': group,
                'kappa': None,
                'total_files': 0
            })
            continue

        labels1 = [group_first_labels[file] for file in sorted_group_files]
        labels2 = [group_second_labels[file] for file in sorted_group_files]

        group_kappa = cohen_kappa_score(labels1, labels2)

        group_results.append({
            'group': group,
            'kappa': group_kappa,
            'total_files': len(sorted_group_files)
        })

    return group_results


def get_strict_file_subset(first_labels):
    """
    严格模式下筛选文件子集：
    - 同名文件在 harmless 和 harmful 中都存在；
    - 两个 json 的 label 字段，一个是 0，一个是 1；
    - 满足条件时，这两个文件都返回。
    """
    # 结构：basename -> {'harmless': (path, label), 'harmful': (path, label)}
    index = {}

    for path, label in first_labels.items():
        basename = os.path.basename(path)
        # 判断路径中是 harmless 还是 harmful
        if os.path.sep + 'harmless' + os.path.sep in path:
            kind = 'harmless'
        elif os.path.sep + 'harmful' + os.path.sep in path:
            kind = 'harmful'
        else:
            # 既不在 harmless 也不在 harmful，跳过
            continue

        if basename not in index:
            index[basename] = {}
        index[basename][kind] = (path, label)

    strict_paths = []

    for basename, info in index.items():
        if 'harmless' in info and 'harmful' in info:
            _, label_harmless = info['harmless']
            _, label_harmful = info['harmful']

            # 只有一边是0、一边是1时才算
            if {label_harmless, label_harmful} == {0, 1}:
                strict_paths.append(info['harmless'][0])
                strict_paths.append(info['harmful'][0])

    # 为了稳定性，按文件名中的数字排序
    def sort_key(p):
        name = os.path.basename(p)
        try:
            return int(name.split('.')[0])
        except ValueError:
            return name

    strict_paths.sort(key=sort_key)
    print("Strict mode: files selected:", len(strict_paths))
    return strict_paths


def main():
    # 获取项目根目录
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from config.paths import DATA_ROOT, REPO_ROOT
    project_root = str(DATA_ROOT)

    # 加载JSON标签
    first_labels, second_labels = load_json_files(project_root)

    # 读取分组信息
    divide_work_path = str(REPO_ROOT / 'pipeline' / 'divide_work.txt')
    groups = read_divide_work(divide_work_path)

    # ---------------------------
    # 不严格模式（原有逻辑，不改）
    # ---------------------------
    total_kappa, all_files = calculate_kappa(first_labels, second_labels)
    group_results = calculate_group_kappa(first_labels, second_labels, groups)

    # ---------------------------
    # 严格模式：基于成对文件 + label(0,1) 条件筛选
    # ---------------------------
    strict_files = get_strict_file_subset(first_labels)

    if len(strict_files) > 0:
        total_kappa_strict, strict_files_used = calculate_kappa(
            first_labels,
            second_labels,
            file_subset=strict_files
        )
        group_results_strict = calculate_group_kappa(
            first_labels,
            second_labels,
            groups,
            file_subset=strict_files
        )
    else:
        total_kappa_strict = None
        strict_files_used = []
        group_results_strict = []

    # 输出总体Kappa值
    print(f"总体Kappa值(不严格): {total_kappa:.4f}")
    print(f"总文件数(不严格): {len(all_files)}")
    if total_kappa_strict is not None:
        print(f"总体Kappa值(严格): {total_kappa_strict:.4f}")
        print(f"总文件数(严格): {len(strict_files_used)}")
    else:
        print("严格模式下没有满足条件的文件对。")

    # 保存结果到CSV
    output_path = os.path.join(project_root, 'construct', 'dataset', 'kappa_with_AI_results.csv')
    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['type', 'group_start', 'group_end', 'kappa', 'total_files']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()

        # 不严格模式 - 总体
        writer.writerow({
            'type': 'total',
            'group_start': None,
            'group_end': None,
            'kappa': total_kappa,
            'total_files': len(all_files)
        })

        # 不严格模式 - 分组
        for result in group_results:
            writer.writerow({
                'type': 'group',
                'group_start': result['group'][0],
                'group_end': result['group'][1],
                'kappa': result['kappa'],
                'total_files': result['total_files']
            })

        # 严格模式 - 总体（追加，不覆盖）
        writer.writerow({
            'type': 'total_strict',
            'group_start': None,
            'group_end': None,
            'kappa': total_kappa_strict,
            'total_files': len(strict_files_used)
        })

        # 严格模式 - 分组（追加，不覆盖）
        for result in group_results_strict:
            writer.writerow({
                'type': 'group_strict',
                'group_start': result['group'][0],
                'group_end': result['group'][1],
                'kappa': result['kappa'],
                'total_files': result['total_files']
            })

    print(f"Kappa值结果已保存到 {output_path}")


if __name__ == '__main__':
    main()