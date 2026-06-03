#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Add Meme Text Script
为restructured目录中的JSON文件添加text字段
从raw_message目录中读取对应的文本内容
"""

import json
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


def construct_raw_message_filename(path_value):
    """根据path字段构造raw_message文件名"""
    if not path_value:
        return None

    # 如果path已经是.jpg结尾，直接添加.json
    if path_value.endswith('.jpg'):
        return f"{path_value}.json"
    else:
        # 如果不是，先添加.jpg再添加.json
        return f"{path_value}.jpg.json"


def process_single_file(json_file, raw_message_dir):
    """处理单个JSON文件，添加text字段"""
    # 加载当前文件
    data = load_json_file(json_file)
    if data is None:
        return False, "加载文件失败"

    # 检查是否已有text字段
    if 'text' in data:
        print(f"  跳过: {json_file.name} (已有text字段)")
        return True, "已有text字段"

    # 获取path字段
    path_value = data.get('path')
    if not path_value:
        print(f"  错误: {json_file.name} 缺少path字段")
        return False, "缺少path字段"

    # 构造raw_message文件名
    raw_message_filename = construct_raw_message_filename(path_value)
    if not raw_message_filename:
        print(f"  错误: {json_file.name} 无法构造raw_message文件名")
        return False, "无法构造文件名"

    # 查找对应的raw_message文件
    raw_message_file = raw_message_dir / raw_message_filename

    if not raw_message_file.exists():
        print(f"  警告: {json_file.name} 对应的raw_message文件不存在: {raw_message_filename}")
        return False, "raw_message文件不存在"

    # 加载raw_message文件
    raw_data = load_json_file(raw_message_file)
    if raw_data is None:
        print(f"  错误: {json_file.name} 无法加载raw_message文件: {raw_message_filename}")
        return False, "无法加载raw_message文件"

    # 获取text字段
    text_content = raw_data.get('text', '')

    # 添加text字段到当前数据
    data['text'] = text_content

    # 保存更新后的文件
    if save_json_file(json_file, data):
        print(f"  ✓ 已更新: {json_file.name} (添加text字段)")
        return True, "成功添加text字段"
    else:
        print(f"  ✗ 保存失败: {json_file.name}")
        return False, "保存文件失败"


def main():
    # 定义路径
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from config.paths import construct_dataset, random_dataset

    restructured_dir = construct_dataset("restructured")
    raw_message_dir = random_dataset("raw_message")

    print("=== 添加Meme文本字段工具 ===")
    print("为restructured目录中的JSON文件添加text字段")

    # 检查目录是否存在
    if not restructured_dir.exists():
        print(f"错误：restructured目录不存在 {restructured_dir}")
        return

    if not raw_message_dir.exists():
        print(f"错误：raw_message目录不存在 {raw_message_dir}")
        return

    print(f"\n源目录: {restructured_dir}")
    print(f"文本源目录: {raw_message_dir}")

    # 获取所有JSON文件
    json_files = list(restructured_dir.glob("*.json"))

    if not json_files:
        print("在restructured目录中没有找到JSON文件")
        return

    print(f"\n找到 {len(json_files)} 个JSON文件需要处理")

    # 统计变量
    total_files = len(json_files)
    success_count = 0
    skip_count = 0
    error_count = 0

    # 错误统计
    error_reasons = {
        "加载文件失败": 0,
        "缺少path字段": 0,
        "无法构造文件名": 0,
        "raw_message文件不存在": 0,
        "无法加载raw_message文件": 0,
        "保存文件失败": 0
    }

    missing_raw_files = []

    print(f"\n=== 开始处理文件 ===")

    # 处理每个文件
    for i, json_file in enumerate(sorted(json_files), 1):
        print(f"[{i}/{total_files}] 处理: {json_file.name}")

        success, reason = process_single_file(json_file, raw_message_dir)

        if success:
            if reason == "已有text字段":
                skip_count += 1
            else:
                success_count += 1
        else:
            error_count += 1
            if reason in error_reasons:
                error_reasons[reason] += 1

            # 记录缺失的raw_message文件
            if reason == "raw_message文件不存在":
                data = load_json_file(json_file)
                if data and 'path' in data:
                    path_value = data['path']
                    raw_filename = construct_raw_message_filename(path_value)
                    if raw_filename:
                        missing_raw_files.append(raw_filename)

    # 输出统计结果
    print(f"\n=== 处理完成 ===")
    print(f"总文件数: {total_files}")
    print(f"成功添加text字段: {success_count}")
    print(f"跳过 (已有text字段): {skip_count}")
    print(f"处理失败: {error_count}")

    if error_count > 0:
        print(f"\n=== 错误详情 ===")
        for reason, count in error_reasons.items():
            if count > 0:
                print(f"{reason}: {count}")

        # 显示缺失的raw_message文件
        if missing_raw_files:
            print(f"\n缺失的raw_message文件 ({len(missing_raw_files)} 个):")
            # 显示前10个，避免输出过多
            for i, filename in enumerate(sorted(set(missing_raw_files))[:10]):
                print(f"  - {filename}")
            if len(set(missing_raw_files)) > 10:
                print(f"  ... 还有 {len(set(missing_raw_files)) - 10} 个文件")

    # 验证结果
    print(f"\n=== 验证结果 ===")
    updated_files = []
    files_with_text = 0
    files_without_text = 0

    for json_file in json_files:
        data = load_json_file(json_file)
        if data:
            if 'text' in data:
                files_with_text += 1
                text_length = len(data['text']) if data['text'] else 0
                if text_length > 0:
                    updated_files.append({
                        'name': json_file.name,
                        'text_length': text_length,
                        'text_preview': data['text'][:50] + ('...' if len(data['text']) > 50 else '')
                    })
            else:
                files_without_text += 1

    print(f"有text字段的文件: {files_with_text}")
    print(f"无text字段的文件: {files_without_text}")

    # 显示一些示例
    if updated_files:
        print(f"\n=== text字段示例 ===")
        for i, file_info in enumerate(sorted(updated_files, key=lambda x: x['name'])[:5]):
            print(f"{file_info['name']} (长度: {file_info['text_length']})")
            print(f"  内容预览: {file_info['text_preview']}")

    if success_count > 0:
        print(f"\n✅ 成功为 {success_count} 个文件添加了text字段")

    if error_count == 0:
        print(" 所有文件处理完成，没有错误！")
    else:
        print(f"  {error_count} 个文件处理时遇到问题")


if __name__ == "__main__":
    main()