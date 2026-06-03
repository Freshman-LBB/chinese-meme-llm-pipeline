import os
import json
import csv
import numpy as np
import re


def weighted_kappa(y_true, y_pred, weights='linear'):
    """
    计算两个标注者之间的加权 Kappa

    weights:
        - 'linear'    : 传统线性权重（差 1 的比差 2 的更相近）
        - 'quadratic' : 传统二次权重
        - 'custom'    : 自定义 0/1/2 之间的相似度：
                        0 与 2 比 0 与 1 更相似，
                        2 与 0 比 2 与 1 更相似。
    """
    # 确保输入是整数列表
    y_true = [int(x) for x in y_true]
    y_pred = [int(x) for x in y_pred]

    # 使用“出现过的类 + 0/1/2”确定全集，用于编码到索引
    classes = sorted(list(set(y_true + y_pred + [0, 1, 2])))
    n_classes = len(classes)

    # 手动构建混淆矩阵
    label_to_ind = {l: i for i, l in enumerate(classes)}
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[label_to_ind[t], label_to_ind[p]] += 1

    # 构建权重矩阵 w[i,j] (相似度，1=完全一致，0=完全不相似)
    w = np.zeros((n_classes, n_classes), dtype=float)

    if weights == 'linear':
        # 只有一个类别时，视为完全一致
        if n_classes == 1:
            return 1.0
        for i in range(n_classes):
            for j in range(n_classes):
                w[i, j] = 1 - abs(i - j) / (n_classes - 1)

    elif weights == 'quadratic':
        if n_classes == 1:
            return 1.0
        for i in range(n_classes):
            for j in range(n_classes):
                w[i, j] = 1 - (abs(i - j) / (n_classes - 1)) ** 2

    elif weights == 'custom':
        # 自定义逻辑针对真实标签 0、1、2，映射到当前 classes 的索引再赋值

        # 先设对角线为 1（完全一致）
        for i in range(n_classes):
            w[i, i] = 1.0

        # 定义三类之间的相似度（可根据业务需求调整数值）
        # 要满足：
        #   w(0,2) > w(0,1)
        #   w(0,2) > w(1,2)
        custom_sim = {
            (0, 1): 0.3,
            (1, 0): 0.3,
            (1, 2): 0.3,
            (2, 1): 0.3,
            (0, 2): 0.7,
            (2, 0): 0.7,
        }

        for (a, b), sim in custom_sim.items():
            if a in label_to_ind and b in label_to_ind:
                ia = label_to_ind[a]
                ib = label_to_ind[b]
                w[ia, ib] = sim
                w[ib, ia] = sim  # 再确保对称

    else:
        raise ValueError("Unsupported weights type")

    total = np.sum(cm)
    if total == 0:
        return 1.0

    # 观测一致性
    obs_agreement = np.sum(w * cm) / total

    # 期望一致性
    sum0 = cm.sum(axis=0)
    sum1 = cm.sum(axis=1)
    expected_agreement = np.sum(w * np.outer(sum1, sum0) / total) / total

    if 1 - expected_agreement == 0:
        return 1.0 if obs_agreement == 1.0 else 0.0

    kappa = (obs_agreement - expected_agreement) / (1 - expected_agreement)
    return kappa


def calculate_fleiss_kappa(data_matrix):
    """
    计算 Fleiss' Kappa (适用于3个及以上标注者，无权重)
    :param data_matrix: N行M列的矩阵，N是样本数，M是标注者数，值为类别
    """
    data_matrix = np.array(data_matrix)
    N, k = data_matrix.shape  # N个样本, k个标注者(3)

    # 统计每个样本中每个类别被标注的次数
    # 假设类别是 0, 1, 2
    categories = [0, 1, 2]
    n_categories = len(categories)

    # 构建计数矩阵 (N x n_categories)
    counts = np.zeros((N, n_categories))
    for i in range(N):
        for j in range(k):
            val = int(data_matrix[i, j])
            if val in categories:
                counts[i, val] += 1

    # 计算 P_i (每个样本的一致性程度)
    # Pi = (1 / (k * (k-1))) * (Sum(n_ij^2) - k)
    P_i = (np.sum(counts ** 2, axis=1) - k) / (k * (k - 1))
    P_bar = np.mean(P_i)

    # 计算 P_e (随机一致性概率)
    p_j = np.sum(counts, axis=0) / (N * k)
    P_e = np.sum(p_j ** 2)

    if P_e == 1:
        return 1.0

    kappa = (P_bar - P_e) / (1 - P_e)
    return kappa


def calculate_avg_pairwise_kappa(labels_list, weight_type='custom'):
    """
    计算平均成对加权 Kappa (适用于3个标注者)
    labels_list: [[r1_doc1, r2_doc1, r3_doc1], [r1_doc2, ...]]
    """
    n_raters = 3
    kappas = []

    # 转置矩阵，变成 [rater1_all_labels, rater2_all_labels, rater3_all_labels]
    raters_data = list(zip(*labels_list))

    # 计算成对 Kappa: (1,2), (1,3), (2,3)
    for i in range(n_raters):
        for j in range(i + 1, n_raters):
            k = weighted_kappa(raters_data[i], raters_data[j], weights=weight_type)
            kappas.append(k)

    return np.mean(kappas) if kappas else 0.0


def load_json_files(base_path):
    """
    加载JSON文件，并做一次初筛：
    只保留 data['label'] 在 {0,1,3} 的样本。
    """
    harmless_path = os.path.join(base_path, 'construct', 'dataset', 'harmless')
    harmful_path = os.path.join(base_path, 'construct', 'dataset', 'harmful')

    files_data = {}

    # 处理两个文件夹
    for folder_path in [harmless_path, harmful_path]:
        if not os.path.exists(folder_path):
            continue
        for filename in os.listdir(folder_path):
            if not filename.endswith('.json'):
                continue

            filepath = os.path.join(folder_path, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 只考虑 label 为 0 或 1 或 3 的 JSON
            # 注意：这里 data['label'] 可能是 str/int，最好统一成 int 再判断
            if 'label' not in data:
                continue
            try:
                main_label = int(data['label'])
            except (TypeError, ValueError):
                continue

            if main_label in (0, 1, 3):
                files_data[filepath] = data

    return files_data


def extract_labels(files_data):
    """
    提取三个标注者的标签，并为两种模式分别准备数据。
    额外返回每个文件对应的标注者 key（label_1~label_5），
    方便后面分析哪个标注者表现最差。
    """

    def find_three_labels(data):
        # 在 label_1 ~ label_5 中寻找有效的标注，同时记录是哪个 label_i
        label_keys = [f"label_{i}" for i in range(1, 6)]
        labels = []
        used_keys = []
        for key in label_keys:
            if key in data and data[key] is not None:
                labels.append(int(data[key]))
                used_keys.append(key)
        # 只有当“恰好”有 3 个标注时才使用
        return (labels, used_keys) if len(labels) == 3 else (None, None)

    filtered_data_mode1 = {}  # 模式1：处理后的 0/1 标签，用于 Fleiss kappa
    filtered_data_mode2 = {}  # 模式2：原始 0/1/2 标签，用于加权 kappa
    rater_keys_map = {}       # 每个文件对应的 ['label_1', 'label_3', 'label_5'] 之类

    for filepath, data in files_data.items():
        labels, used_keys = find_three_labels(data)
        if labels is None:
            continue  # 没有恰好3个有效标注

        # 记录这个文件用的是哪几个 label_i
        rater_keys_map[filepath] = used_keys

        # ---------- 模式2：原始标签直接保留 ----------
        # 注意用拷贝，避免后面对 labels 的修改影响到这里
        filtered_data_mode2[filepath] = labels.copy()  # 0/1/2 原样

        # ---------- 模式1：按照规则处理 2 ----------
        # 统计频次（基于原始 labels）
        counts = {}
        for l in labels:
            counts[l] = counts.get(l, 0) + 1

        # 若 2 出现两次或以上，则该样本不用于模式1
        if counts.get(2, 0) >= 2:
            continue

        if counts.get(2, 0) == 1:
            # 仅一个人标2，尝试看另外两人是否形成多数类 0 或 1
            non_two_labels = [l for l in labels if l != 2]
            # 例如 [0,0,2] 或 [1,1,2]
            if len(set(non_two_labels)) == 1 and non_two_labels[0] in (0, 1):
                majority = non_two_labels[0]
                labels_mode1 = [majority, majority, majority]
            else:
                # 如 [0,1,2] 之类，无明确多数类 -> 丢弃
                continue
        else:
            # 根本没有 2，直接用原来的标签，但要确保只含 0/1
            if any(l not in (0, 1) for l in labels):
                # 若包含 3 或其它值，则不用于 mode1（二分类专用）
                continue
            labels_mode1 = labels.copy()

        filtered_data_mode1[filepath] = labels_mode1

    return filtered_data_mode1, filtered_data_mode2, rater_keys_map

def analyze_group_raters(group_data, rater_keys_map, mode_name):
    """
    对某一组（group_data）和某一种模式（fleiss / weighted），
    计算参与的各个标注者（label_1~label_5）之间的成对 kappa，
    返回平均 pairwise kappa 最低的那个标注者。

    group_data: {filepath: [l1, l2, l3]}  # 针对某一模式筛完后的数据
    rater_keys_map: {filepath: ['label_1', 'label_3', 'label_5']}
    mode_name: 'fleiss' or 'weighted'
    """
    if not group_data:
        return None

    # 收集每个标注者在该组中的标签：rater -> {filepath: label}
    rater_labels = {}
    for filepath, labels in group_data.items():
        if filepath not in rater_keys_map:
            continue
        keys = rater_keys_map[filepath]  # 例如 ['label_1', 'label_3', 'label_5']
        # keys 和 labels 一一对应
        for idx, r_key in enumerate(keys):
            if idx >= len(labels):
                continue
            rater_labels.setdefault(r_key, {})[filepath] = labels[idx]

    raters = list(rater_labels.keys())
    if len(raters) < 2:
        # 只有一个标注者或者没有，没法算 pairwise
        return None

    # 先算出所有成对的 kappa
    pairwise_kappas = {r: [] for r in raters}

    for i in range(len(raters)):
        for j in range(i + 1, len(raters)):
            r_a = raters[i]
            r_b = raters[j]

            # 只在两人都标注了的样本上算 kappa
            files_ij = set(rater_labels[r_a].keys()) & set(rater_labels[r_b].keys())
            if len(files_ij) < 2:
                # 样本太少，跳过这对
                continue

            y_a = [rater_labels[r_a][fp] for fp in files_ij]
            y_b = [rater_labels[r_b][fp] for fp in files_ij]

            if mode_name == 'weighted':
                k = weighted_kappa(y_a, y_b, weights='custom')
            else:  # fleiss（二分类），这里用线性权重的 Cohen kappa
                k = weighted_kappa(y_a, y_b, weights='linear')

            pairwise_kappas[r_a].append(k)
            pairwise_kappas[r_b].append(k)

    # 对每个标注者，取其与他人 kappa 的平均值
    mean_kappa_per_rater = {}
    for r, vals in pairwise_kappas.items():
        if vals:
            mean_kappa_per_rater[r] = float(np.mean(vals))

    if not mean_kappa_per_rater:
        return None

    # 平均 kappa 最小的那个视为“表现最差”
    worst_rater = min(mean_kappa_per_rater, key=mean_kappa_per_rater.get)
    return {
        'worst_rater': worst_rater,
        'worst_rater_mean_kappa': mean_kappa_per_rater[worst_rater],
        'mean_kappa_per_rater': mean_kappa_per_rater,
    }

def process_kappa_calculation(files_data, groups=None):
    """
    计算Kappa值，同时在每个分组中标记：
    哪个标注者（label_1~label_5）平均 pairwise kappa 最低。
    """
    results = {}

    def calculate_metrics_for_dataset(data_dict, mode):
        """
        data_dict: {filepath: [l1, l2, l3]}
        """
        if not data_dict:
            return 0.0, 0

        all_labels_matrix = list(data_dict.values())

        if mode == 'fleiss':
            kappa = calculate_fleiss_kappa(all_labels_matrix)
        elif mode == 'weighted':
            kappa = calculate_avg_pairwise_kappa(all_labels_matrix, weight_type='custom')
        else:
            kappa = 0.0

        return kappa, len(data_dict)

    # 提取数据（注意这里多了 rater_keys_map）
    mode1_data, mode2_data, rater_keys_map = extract_labels(files_data)

    # 计算不同模式
    for mode_name in ['fleiss', 'weighted']:
        results[mode_name] = {}

        # 选择对应的数据集
        current_data = mode1_data if mode_name == 'fleiss' else mode2_data

        # 情况1：分组计算
        if groups:
            results[mode_name]['groups'] = []
            for group in groups:
                # 筛选属于该组的文件
                group_data = {
                    k: v for k, v in current_data.items()
                    if int(os.path.basename(k).split('.')[0]) in range(group[0], group[1] + 1)
                }

                kappa, total_files = calculate_metrics_for_dataset(group_data, mode_name)

                # 计算该组中“表现最差”的标注者
                worst_info = analyze_group_raters(group_data, rater_keys_map, mode_name)

                if worst_info is not None:
                    worst_rater = worst_info['worst_rater']
                    worst_rater_mean_kappa = worst_info['worst_rater_mean_kappa']
                else:
                    worst_rater = None
                    worst_rater_mean_kappa = None

                results[mode_name]['groups'].append({
                    'group': group,
                    'kappa': kappa,
                    'total_files': total_files,
                    'worst_rater': worst_rater,
                    'worst_rater_mean_kappa': worst_rater_mean_kappa,
                })

        # 情况2：全局计算
        kappa_total, total_files_total = calculate_metrics_for_dataset(current_data, mode_name)
        results[mode_name]['total'] = {
            'kappa': kappa_total,
            'total_files': total_files_total
        }

    return results

def save_results_to_csv(results, output_path):
    """
    将结果保存到CSV，包括：
    - 每种指标在全体样本上的 kappa
    - 每种指标在各个分组上的 kappa
    - 每个分组中平均 pairwise kappa 最低的标注者
    """
    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)

        # 写入表头（多了两列）
        writer.writerow([
            'Metric Type', 'Scope', 'Group Range',
            'Kappa Value', 'File Count',
            'Worst Rater', 'Worst Rater Mean Pairwise Kappa'
        ])

        # 映射名字以便阅读
        metric_name_map = {
            'fleiss': 'Fleiss Kappa (Binary 0/1)',
            'weighted': 'Avg Pairwise Weighted Kappa (All 0/1/2)'
        }

        for mode, mode_data in results.items():
            display_mode = metric_name_map.get(mode, mode)

            # 写入总体结果（不分组，这里不写 worst rater）
            if 'total' in mode_data:
                total = mode_data['total']
                writer.writerow([
                    display_mode, 'Total All Files', 'All',
                    f"{total['kappa']:.4f}", total['total_files'],
                    '', ''  # 不分析总体的最差标注者，这两列留空
                ])

            # 写入分组结果
            if 'groups' in mode_data:
                for g in mode_data['groups']:
                    range_str = f"{g['group'][0]}-{g['group'][1]}"

                    worst_rater = g.get('worst_rater')
                    worst_rater_mean = g.get('worst_rater_mean_kappa')

                    writer.writerow([
                        display_mode,
                        'Group',
                        range_str,
                        f"{g['kappa']:.4f}",
                        g['total_files'],
                        worst_rater if worst_rater is not None else '',
                        f"{worst_rater_mean:.4f}" if worst_rater_mean is not None else ''
                    ])


def main():
    # 获取项目根目录
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from config.paths import DATA_ROOT, REPO_ROOT
    project_root = str(DATA_ROOT)

    # 加载JSON文件
    print("正在加载数据...")
    files_data = load_json_files(project_root)
    print(f"加载了 {len(files_data)} 个文件")

    # 读取分组信息
    groups = []
    divide_work_path = str(REPO_ROOT / 'pipeline' / 'divide_work.txt')
    if os.path.exists(divide_work_path):
        with open(divide_work_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # 使用正则从形如 "[58, 1950] 50" 里提取 58 和 1950
                m = re.search(r'\[(\d+)\s*,\s*(\d+)\]', line)
                if m:
                    start = int(m.group(1))
                    end = int(m.group(2))
                    groups.append([start, end])
    else:
        print("未找到分组文件，将跳过分组计算。")
    # 计算Kappa值
    print("正在计算一致性...")
    results = process_kappa_calculation(files_data, groups=groups)

    # 保存结果
    output_path = os.path.join(project_root, 'construct', 'dataset', 'kappa_3_labelers_results.csv')
    save_results_to_csv(results, output_path)

    # 打印结果
    print(f"结果已保存到: {output_path}")


if __name__ == '__main__':
    main()