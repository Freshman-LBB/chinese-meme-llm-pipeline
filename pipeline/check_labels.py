import json
import csv
from pathlib import Path
from collections import defaultdict
import pandas as pd
import os


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


def calculate_metrics(comparison_data):
    """计算混淆矩阵和性能指标，返回所有值"""
    if not comparison_data:
        return 0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0

    stats = comparison_data['statistics']
    comparison_results = comparison_data['comparison_results']

    # 计算混淆矩阵
    tp = 0  # True Positive: LLM=1, Raw=1
    fp = 0  # False Positive: LLM=1, Raw=0
    tn = 0  # True Negative: LLM=0, Raw=0
    fn = 0  # False Negative: LLM=0, Raw=1

    for result in comparison_results:
        llm_label = result['llm_label']
        raw_label = result['raw_label']

        # 只计算有效的比较结果（raw_label不为None且不为2）
        if raw_label is not None and raw_label != 2:
            if llm_label == 1 and raw_label == 1:
                tp += 1
            elif llm_label == 1 and raw_label == 0:
                fp += 1
            elif llm_label == 0 and raw_label == 0:
                tn += 1
            elif llm_label == 0 and raw_label == 1:
                fn += 1

    print(f"\n=== 混淆矩阵 ===")
    print(f"True Positive (TP): {tp}")
    print(f"False Positive (FP): {fp}")
    print(f"True Negative (TN): {tn}")
    print(f"False Negative (FN): {fn}")

    # 计算指标
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0

    print(f"\n=== 性能指标 ===")
    print(f"精确率 (Precision): {precision:.2%}")
    print(f"召回率 (Recall): {recall:.2%}")
    print(f"F1分数: {f1_score:.2%}")
    print(f"准确率 (Accuracy): {accuracy:.2%}")

    # 返回混淆矩阵和性能指标
    return tp, fp, tn, fn, precision, recall, f1_score, accuracy


def process_llm_files(llm_dir):
    """
    处理LLM_with_content目录中的JSON文件
    为所有JSON文件添加label标签
    """
    if not llm_dir.exists():
        print(f"错误：目录不存在 {llm_dir}")
        return []

    json_files = list(llm_dir.glob("*.json"))
    print(f"找到 {len(json_files)} 个JSON文件需要处理")

    processed_files = []
    modified_count = 0

    for json_file in json_files:
        data = load_json_file(json_file)
        if data is None:
            continue

        modified = False

        # 检查是否已有label字段
        if 'label' in data:
            print(f"跳过: {json_file.name} (已有label字段，值为{data['label']})")
        else:
            # 没有label字段，需要根据content添加
            if 'content' in data:
                content = data['content']
                if content == "无害":
                    data['label'] = 0
                elif content == "有害":
                    data['label'] = 1
                else:
                    print(f"警告: {json_file.name} content字段值异常: '{content}' -> 设置为2")
                    data['label'] = 2  # 异常内容
            else:
                print(f"警告: {json_file.name} 没有content字段，设置label为2")
                data['label'] = 2  # 异常内容

            modified = True

        # 保存修改后的文件
        if modified:
            if save_json_file(json_file, data):
                modified_count += 1
                print(f"✓ 已更新: {json_file.name} (label={data['label']})")

        processed_files.append({
            'file_path': json_file,
            'file_name': json_file.name,
            'label': data.get('label', 2),
            'content': data.get('content', ''),
            'data': data
        })

    print(f"\n处理完成，修改了 {modified_count} 个文件")
    return processed_files


def analyze_labels(processed_files):
    """分析标签分布"""
    label_counts = defaultdict(int)
    files_with_label_2 = []

    for file_info in processed_files:
        label = file_info['label']
        label_counts[label] += 1

        if label == 2:
            files_with_label_2.append(file_info['file_name'])

    print(f"\n=== 标签分布统计 ===")
    print(f"标签 0 (无害): {label_counts[0]} 个文件")
    print(f"标签 1 (有害): {label_counts[1]} 个文件")
    print(f"标签 2 (异常): {label_counts[2]} 个文件")

    if files_with_label_2:
        print(f"\n标签值为2的文件列表:")
        for filename in files_with_label_2:
            print(f"  - {filename}")

    return label_counts, files_with_label_2


def compare_with_raw_message(processed_files, raw_message_dir,storage_file):
    """与humman-anno/dataset/restructured目录中的同名文件进行比较"""
    if not raw_message_dir.exists():
        print(f"错误：restructured目录不存在 {raw_message_dir}")
        return None

    print(f"\n=== 与restructured目录比较 ===")

    comparison_results = []
    same_count = 0
    different_count = 0
    not_found_count = 0

    llm_0_inconsistent = []  # LLM为0且不一致的文件
    llm_1_inconsistent = []  # LLM为1且不一致的文件

    for file_info in processed_files:
        llm_filename = file_info['file_name']
        llm_label = file_info['label']

        # 直接使用原始文件名搜索对应的restructured文件
        raw_file_path = raw_message_dir / llm_filename

        if raw_file_path.exists():
            raw_data = load_json_file(raw_file_path)
            if raw_data is not None:
                if storage_file == "without":
                    raw_label = raw_data.get('raw_label', None)
                else: raw_label = raw_data.get('label', None)
                comparison_result = {
                    'llm_file': llm_filename,
                    'raw_file': llm_filename,  # 同名文件
                    'llm_label': llm_label,
                    'raw_label': raw_label,
                    'match': llm_label == raw_label
                }
                comparison_results.append(comparison_result)

                if llm_label == raw_label:
                    same_count += 1
                else:
                    different_count += 1
                    print(f"不一致: {llm_filename} - LLM:{llm_label}, Raw:{raw_label}")

                    # 记录LLM标签不一致的情况（仅当raw_label为有效值0或1时）
                    if raw_label in [0, 1]:
                        if llm_label == 0:
                            llm_0_inconsistent.append(llm_filename)
                        elif llm_label == 1:
                            llm_1_inconsistent.append(llm_filename)
            else:
                not_found_count += 1
                comparison_results.append({
                    'llm_file': llm_filename,
                    'raw_file': llm_filename,
                    'llm_label': llm_label,
                    'raw_label': None,
                    'match': False
                })
        else:
            not_found_count += 1
            print(f"未找到对应文件: {llm_filename}")
            comparison_results.append({
                'llm_file': llm_filename,
                'raw_file': llm_filename,
                'llm_label': llm_label,
                'raw_label': None,
                'match': False
            })

    total_compared = same_count + different_count
    print(f"\n比较结果统计:")
    print(f"总比较文件数: {total_compared}")
    print(f"标签一致: {same_count}")
    print(f"标签不一致: {different_count}")
    print(f"未找到对应文件: {not_found_count}")

    if llm_0_inconsistent:
        print(f"\nLLM标注为0且不一致的文件数量: {len(llm_0_inconsistent)}")
        print("文件列表:")
        for filename in llm_0_inconsistent:
            print(f"  - {filename}")

    if llm_1_inconsistent:
        print(f"\nLLM标注为1且不一致的文件数量: {len(llm_1_inconsistent)}")
        print("文件列表:")
        for filename in llm_1_inconsistent:
            print(f"  - {filename}")

    return {
        'comparison_results': comparison_results,
        'statistics': {
            'total_compared': total_compared,
            'same_count': same_count,
            'different_count': different_count,
            'not_found_count': not_found_count,
            'llm_0_inconsistent': llm_0_inconsistent,
            'llm_1_inconsistent': llm_1_inconsistent
        }
    }


def save_summary_to_csv(processed_files, label_counts, files_with_label_2, comparison_data, output_dir,storage_file):
    """保存汇总结果到CSV文件，包含混淆矩阵和性能指标"""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 计算混淆矩阵和性能指标
    tp, fp, tn, fn, precision, recall, f1_score, accuracy = calculate_metrics(comparison_data)

    # 准备数据
    total_files = len(processed_files)
    abnormal_count = len(files_with_label_2)

    if comparison_data:
        stats = comparison_data['statistics']
        consistent_count = stats['same_count']
        inconsistent_0_count = len(stats['llm_0_inconsistent'])
        inconsistent_0_list = '; '.join(stats['llm_0_inconsistent'])
        inconsistent_1_count = len(stats['llm_1_inconsistent'])
        inconsistent_1_list = '; '.join(stats['llm_1_inconsistent'])
        total_compared = stats['total_compared']
        not_found_count = stats['not_found_count']
    else:
        consistent_count = 0
        inconsistent_0_count = 0
        inconsistent_0_list = ''
        inconsistent_1_count = 0
        inconsistent_1_list = ''
        total_compared = 0
        not_found_count = 0

    # 删除原有CSV文件（如果存在）
    summary_file=""
    if storage_file=="with":
        summary_file = output_dir / 'analysis_summary(with_content).csv'
    elif storage_file=="without":
        summary_file = output_dir / 'analysis_summary(without_content).csv'
    if summary_file.exists():
        os.remove(summary_file)
        print(f"已删除原有CSV文件: {summary_file}")

    # 保存到CSV文件
    with open(summary_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)

        # 写入表头 - 添加混淆矩阵列
        writer.writerow([
            '总文件数目',
            '异常标签数目(label=2)',
            '成功比较数目',
            '一致数目',
            '不一致数目(LLM=无害)',
            '不一致文件列表(LLM=无害)',
            '不一致数目(LLM=有害)',
            '不一致文件列表(LLM=有害)',
            '未找到文件数目',
            'TP',
            'FP',
            'TN',
            'FN',
            '精确率',
            '召回率',
            'F1分数',
            '准确率'
        ])

        # 写入数据 - 添加混淆矩阵值
        writer.writerow([
            total_files,
            abnormal_count,
            total_compared,
            consistent_count,
            inconsistent_0_count,
            inconsistent_0_list,
            inconsistent_1_count,
            inconsistent_1_list,
            not_found_count,
            tp,
            fp,
            tn,
            fn,
            f'{precision:.4f}',
            f'{recall:.4f}',
            f'{f1_score:.4f}',
            f'{accuracy:.4f}'
        ])

    print(f"\n汇总结果已保存到: {summary_file}")
    print(f"CSV包含混淆矩阵和性能指标:")
    print(f"- TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}")
    print(f"- 精确率: {precision:.4f}")
    print(f"- 召回率: {recall:.4f}")
    print(f"- F1分数: {f1_score:.4f}")
    print(f"- 准确率: {accuracy:.4f}")


def main(storage_file):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from config.paths import construct_dataset

    llm_dir = ""
    if storage_file == "with":
        llm_dir = construct_dataset("LLM_with_content")
    elif storage_file == "without":
        llm_dir = construct_dataset("LLM_without_content")
    raw_message_dir = construct_dataset("restructured")
    output_dir = construct_dataset("label_analysis_results")

    print("=== LLM标签检查和处理工具 ===")
    print(f"LLM目录: {llm_dir}")
    print(f"对比目录: {raw_message_dir}")
    print(f"输出目录: {output_dir}")

    # 第一步：处理目录中的文件，添加label标签
    print("\n步骤1: 为LLM_with_content目录中的JSON文件添加label标签")
    processed_files = process_llm_files(llm_dir)

    if not processed_files:
        print("没有找到可处理的文件")
        return

    # 第二步：分析标签分布
    print("\n步骤2: 分析标签分布")
    label_counts, files_with_label_2 = analyze_labels(processed_files)

    # 第三步：与restructured目录中的同名文件比较
    print("\n步骤3: 与restructured目录中的同名文件进行比较")
    comparison_data = compare_with_raw_message(processed_files, raw_message_dir,storage_file)

    # 第四步：计算并保存性能指标到CSV
    print("\n步骤4: 计算性能指标并保存到CSV")
    save_summary_to_csv(processed_files, label_counts, files_with_label_2, comparison_data, output_dir,storage_file)

    print("\n=== 处理完成 ===")


if __name__ == "__main__":
    main("with")