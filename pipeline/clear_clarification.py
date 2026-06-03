import os
import json
import re
import shutil


def validate_clarification(content):
    """验证并处理clarification字段"""
    valid_categories = {
        'A': '工作', 'B': '学习生活', 'C': '性相关', 'D': '日常吃喝玩乐',
        'E': '圈内梗', 'F': '社会热点', 'G': '家庭代际', 'H': '外貌/身材',
        'I': '人际情感', 'J': '金钱/消费', 'K': '健康', 'L': '运动',
        'M': '科技/数码', 'N': '娱乐/明星', 'O': '政治/社会性议题',
        'P': '教育', 'Q': '艺术/文创', 'R': '环境/自然',
        'S': '兴趣爱好', 'T': '其他小众'
    }

    # 处理clarification为空或不存在的情况
    if not content or 'clarification' not in content:
        return None

    clarification = content['clarification']

    # 确保返回单字母
    if len(clarification) == 1 and clarification.upper() in valid_categories:
        return clarification.upper()

    # 处理多字符情况，严格限制为单字母
    # 特殊处理T类：只允许T开头，且T后面可以有具体描述
    if clarification[0].upper() == 'T':
        # 如果是T开头，但长度大于1，保留T，舍弃后面的描述
        return clarification

    # 提取第一个大写字母
    match = re.search(r'[A-Z]', clarification)
    if match:
        letter = match.group(0)
        if letter in valid_categories:
            return letter

    return None


def clear_clarification(base_dir):
    """清理clarification目录"""
    # 设置路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    import sys
    from pathlib import Path as _Path
    sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
    from config.paths import random_dataset

    clarification_dir = str(random_dataset("clarification"))

    # 创建备份目录
    backup_dir = os.path.join(clarification_dir, 'backup')
    os.makedirs(backup_dir, exist_ok=True)

    # 记录变更
    modified_files = []
    deleted_files = []
    total_files = 0
    processed_files = 0

    # 遍历目录
    for filename in os.listdir(clarification_dir):
        if filename.endswith('.json'):
            total_files += 1
            file_path = os.path.join(clarification_dir, filename)

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = json.load(f)

                new_category = validate_clarification(content)

                if new_category is None:
                    # 删除文件
                    deleted_files.append(filename)
                    backup_path = os.path.join(backup_dir, filename)
                    shutil.move(file_path, backup_path)
                elif new_category[0].upper() == 'T':
                    # T 类保持不变
                    processed_files += 1
                    continue
                elif len(new_category) != 1:
                    # 如果处理后不是单字母，删除文件
                    deleted_files.append(filename)
                    backup_path = os.path.join(backup_dir, filename)
                    shutil.move(file_path, backup_path)
                else:
                    # 修改文件
                    content['clarification'] = new_category
                    modified_files.append(filename)
                    processed_files += 1

                    # 保存修改
                    with open(file_path, 'w', encoding='utf-8') as f:
                        json.dump(content, f, ensure_ascii=False, indent=2)

            except Exception as e:
                print(f"处理 {filename} 时出错: {e}")
                deleted_files.append(filename)
                backup_path = os.path.join(backup_dir, filename)
                shutil.move(file_path, backup_path)

    # 打印结果
    print("\n清理clarification目录结果:")
    print(f"总计文件：{total_files}")
    print(f"处理成功文件：{processed_files}")
    print(f"删除的文件：{len(deleted_files)}")

    if modified_files:
        print("\n修改的文件列表：")
        print(", ".join(modified_files))

    if deleted_files:
        print("\n删除的文件列表：")
        print(", ".join(deleted_files))


def main():
    clear_clarification(os.path.dirname(os.path.abspath(__file__)))


if __name__ == "__main__":
    main()