#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
脚本作用：
1. 读取三个 label_stats_X.csv 文件。
2. 在“单个 CSV 内”对每个图片(json)分类为 1~4 类：
   前提：一个 json 在一个 CSV 里最多出现 2 次，且一定是
        harmful-files 一次 + harmless-files 一次（或只出现 1 次）。

   - 类别1（动态有害）：
        出现 2 次，且分布在 label=0 和 label=1 的行各一次。
        => total_count = 2, labels = {0, 1}
   - 类别2（严格有害）：
        出现 2 次，且只在 label=1 的行，
        或者一次在 label=1 的行，一次在 label=2 的行。
        => total_count = 2, labels in {{1}, {1, 2}}
   - 类别3（严格无害）：
        出现 2 次，且只在 label=0 的行，
        或者一次在 label=0 的行，一次在 label=2 的行。
        => total_count = 2, labels in {{0}, {0, 2}}
   - 类别4（无法判断）：
        只在 label=0 或 label=1 出现 1 次，
        或者只在 label=2 里出现的（1次或2次）。
        => total_count = 1, 或 (total_count = 2 且 labels = {2})

3. 跨三个 CSV 做全局“去重分类”：
   - 强类别 = {动态有害(1), 严格有害(2), 严格无害(3)}，弱类别 = 无法判断(4)
   - 若某图在任意 CSV 中有 1，则最终类别为 1；
   - 否则：
       * 若强类别集合为 {2} -> 最终 2
       * 若强类别集合为 {3} -> 最终 3
       * 若强类别集合为 {2,3} -> 冲突，不计入任何类
   - 若只有 4（无强类别）-> 最终 4

4. 输出一个汇总 CSV（相对项目根目录）：
   construct/dataset/pick_and_divide_result/pick_and_divide_summary.csv

   表头：
     类别,总数（不去重）,总数（去重）,
     文件名1数量,文件名2数量,文件名3数量,
     文件名1文件,文件名2文件,文件名3文件

   约定：
   - 类别：中文（动态有害 / 严格有害 / 严格无害 / 无法判断）
   - 总数（不去重）：三个 CSV 内该类别计数相加（跨 CSV 不去重）
   - 总数（去重）：按全局规则合并后，该类别的唯一图片数
   - 文件名N数量 (N=1,2,3)：
       格式：a(b)
         a = 第 N 个 CSV 中，该类别的图片数（只对该 CSV 内去重）
         b = 对“总数（去重）”的贡献数量：
             即这张图最终类别为该类，且在第 N 个 CSV 中也被判为该类
   - 文件名N文件：
       第 N 个 CSV 里该类别的所有 jpg 名，用 ';' 分隔，不过滤掉被去重剔除的图片
"""

import csv
import os
from collections import defaultdict
from typing import Dict, List, Set

# 类别编号 -> 中文名称
CATEGORY_NAMES = {
    1: "动态有害",
    2: "严格有害",
    3: "严格无害",
    4: "无法判断",
}


def parse_files_field(field: str) -> List[str]:
    """解析 harmful-files / harmless-files 字段，按逗号拆分。"""
    if not field:
        return []
    parts = field.split(",")
    files = []
    for p in parts:
        p = p.strip()
        if p:
            files.append(p)
    return files


def json_to_jpg_name(filename: str) -> str:
    """将 '13710.jpg.json' 转为 '13710.jpg'。"""
    filename = filename.strip()
    if filename.endswith(".json"):
        return filename[:-5]
    return filename


def collect_file_stats_in_csv(csv_path: str) -> Dict[str, dict]:
    """
    统计单个 CSV 中每个文件的出现情况。

    返回结构:
    {
        "xxx.jpg": {
            "total_count": int,       # 总出现次数
            "harmful_count": int,     # 在 harmful 列出现次数
            "harmless_count": int,    # 在 harmless 列出现次数
            "labels": set()           # 出现过的 label 集合 {0, 1, 2}
        }
    }
    """
    def make_stats():
        return {
            "total_count": 0,
            "harmful_count": 0,
            "harless_count": 0,  # 这里键名写错不影响逻辑，因为不再使用
            "harmless_count": 0,
            "labels": set(),
        }

    file_stats: Dict[str, dict] = defaultdict(make_stats)

    if not os.path.exists(csv_path):
        print(f"[WARN] CSV 文件不存在：{csv_path}")
        return file_stats

    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label_str = (row.get("label") or "").strip()
            # 现在 label 可能为 0 / 1 / 2，其他忽略
            if label_str not in {"0", "1", "2"}:
                continue

            label_val = int(label_str)
            harmful_files = parse_files_field(row.get("harmful-files", ""))
            harmless_files = parse_files_field(row.get("harmless-files", ""))

            # harmful 列
            for jf in harmful_files:
                jpg = json_to_jpg_name(jf)
                file_stats[jpg]["total_count"] += 1
                file_stats[jpg]["harmful_count"] += 1
                file_stats[jpg]["labels"].add(label_val)

            # harmless 列
            for jf in harmless_files:
                jpg = json_to_jpg_name(jf)
                file_stats[jpg]["total_count"] += 1
                file_stats[jpg]["harmless_count"] += 1
                file_stats[jpg]["labels"].add(label_val)

    return file_stats


def classify_files_in_single_csv(file_stats: Dict[str, dict]) -> Dict[str, int]:
    """
    根据单个 CSV 内统计信息，判定每个 jpg 的类别（1~4）。

    规则（单个 CSV 内）：

      若 total_count == 1：
        - 若唯一的 label == 0：  直接丢弃，不计入任何类别
        - 若唯一的 label ∈ {1,2,3}：归为 类别4（无法判断）

      若 total_count == 2：
        - 类别1（动态有害）:
            labels = {0, 1}

        - 类别2（严格有害）:
            labels ∈ {{1}, {1, 2}, {1, 3}}

        - 类别3（严格无害）:
            labels ∈ {{0}, {0, 2}, {0, 3}}

        - 类别4（无法判断）:
            labels ⊆ {2, 3}    # 即 {2}, {3}, 或 {2,3}

      其它情况（total_count > 2 或 label 异常）忽略。
    """
    result: Dict[str, int] = {}

    for jpg, info in file_stats.items():
        total_count: int = info["total_count"]
        labels: Set[int] = info["labels"]

        # 只出现 1 次的情况
        if total_count == 1:
            # 如果唯一的 label 是 0：直接跳过，不分类
            if labels == {0}:
                continue
            # 其它 label(1/2/3)：归为 无法判断(4)
            result[jpg] = 4
            continue

        # 出现 2 次：按 label 组合细分
        if total_count == 2:
            # 动态有害：{0,1}
            if labels == {0, 1}:
                result[jpg] = 1
                continue

            # 严格有害：{1}, {1,2}, {1,3}
            if labels in ({1}, {1, 2}, {1, 3}):
                result[jpg] = 2
                continue

            # 严格无害：{0}, {0,2}, {0,3}
            if labels in ({0}, {0, 2}, {0, 3}):
                result[jpg] = 3
                continue

            # 无法判断：labels 是 {2}, {3}, 或 {2,3}
            if labels.issubset({2, 3}):
                result[jpg] = 4
                continue

        # 其它情况（比如 total_count > 2 或 label 不在 0–3）就忽略，不给类别

    return result


def deduplicate_categories(per_file_categories: Dict[str, List[int]]) -> Dict[str, int]:
    """
    跨三个 CSV 的去重分类。

    输入: jpg -> [cat1, cat2, ...]
    输出: jpg -> 最终类别 (1~4)，若冲突则不返回该 jpg。
    """
    final_category: Dict[str, int] = {}

    for jpg, cat_list in per_file_categories.items():
        unique_cats = set(cat_list)

        # 强类别 1/2/3，弱类别 4
        strong_cats = {c for c in unique_cats if c in {1, 2, 3}}

        chosen = None

        if not strong_cats:
            # 没有强类别，只有 4 -> 最终为 4
            if 4 in unique_cats:
                chosen = 4
        else:
            # 有强类别，按优先级处理
            if 1 in strong_cats:
                chosen = 1
            else:
                if 2 in strong_cats and 3 in strong_cats:
                    # 2 和 3 冲突，且没有 1 -> 不计数
                    chosen = None
                elif 2 in strong_cats:
                    chosen = 2
                elif 3 in strong_cats:
                    chosen = 3

        if chosen is not None:
            final_category[jpg] = chosen

    return final_category


def main():
    # 项目根目录 = 当前脚本的上两级目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(script_dir, "..", ".."))

    csv_rel_paths = [
        os.path.join("construct", "dataset", "label_stats_1.csv"),
        os.path.join("construct", "dataset", "label_stats_2.csv"),
        os.path.join("construct", "dataset", "label_stats_3.csv"),
    ]
    csv_paths = [os.path.join(root_dir, p) for p in csv_rel_paths]

    # per_csv_files[csv_index][category] = set(jpg)
    per_csv_files: List[Dict[int, Set[str]]] = [
        {1: set(), 2: set(), 3: set(), 4: set()},
        {1: set(), 2: set(), 3: set(), 4: set()},
        {1: set(), 2: set(), 3: set(), 4: set()},
    ]

    # per_csv_categories[csv_index] = {jpg: category}
    per_csv_categories: List[Dict[str, int]] = []

    # per_file_categories = {jpg: [cat_from_csv1, cat_from_csv2, ...]}
    per_file_categories: Dict[str, List[int]] = defaultdict(list)

    # 1. 单个 CSV 内分类
    for idx, csv_path in enumerate(csv_paths):
        print(f"[INFO] 正在处理 CSV ({idx+1}/3): {csv_path}")
        file_stats = collect_file_stats_in_csv(csv_path)
        csv_cats = classify_files_in_single_csv(file_stats)

        per_csv_categories.append(csv_cats)

        for jpg, cat in csv_cats.items():
            per_csv_files[idx][cat].add(jpg)
            per_file_categories[jpg].append(cat)

    # 2. 全局去重分类
    final_category_map = deduplicate_categories(per_file_categories)

    # 3. 统计总数（不去重 / 去重）
    nodedupe_total_counts = {1: 0, 2: 0, 3: 0, 4: 0}
    dedup_total_counts = {1: 0, 2: 0, 3: 0, 4: 0}

    # 不去重总数：三个 CSV 内该类别的数量相加
    for idx in range(3):
        for cat in (1, 2, 3, 4):
            nodedupe_total_counts[cat] += len(per_csv_files[idx][cat])

    # 去重总数：最终类别为该类的 jpg 数
    for cat in final_category_map.values():
        dedup_total_counts[cat] += 1

    # 4. 计算每个 CSV 对“去重后总数”的贡献
    per_csv_dedup_contrib: List[Dict[int, int]] = [
        {1: 0, 2: 0, 3: 0, 4: 0},
        {1: 0, 2: 0, 3: 0, 4: 0},
        {1: 0, 2: 0, 3: 0, 4: 0},
    ]

    for idx in range(3):
        current_csv_cats = per_csv_categories[idx]
        for jpg, final_cat in final_category_map.items():
            if current_csv_cats.get(jpg) == final_cat:
                per_csv_dedup_contrib[idx][final_cat] += 1

    # 5. 写汇总 CSV
    output_dir = os.path.join(root_dir, "construct", "dataset", "feature")
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "pick_and_divide_summary.csv")

    headers = [
        "类别",
        "总数（不去重）",
        "总数（去重）",
        "文件名1数量",
        "文件名2数量",
        "文件名3数量",
        "文件名1文件",
        "文件名2文件",
        "文件名3文件",
    ]

    with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)

        # 类别顺序：1,2,3,4
        for cat in (1, 2, 3, 4):
            row = []

            # 类别中文名称
            row.append(CATEGORY_NAMES[cat])

            # 总数（不去重 & 去重）
            row.append(nodedupe_total_counts[cat])
            row.append(dedup_total_counts[cat])

            # 文件名N数量：a(b)
            for idx in range(3):
                a = len(per_csv_files[idx][cat])        # 本 CSV 内该类数量
                b = per_csv_dedup_contrib[idx][cat]     # 对去重后总数贡献
                row.append(f"{a}({b})")

            # 文件名N文件：jpg1;jpg2;...
            for idx in range(3):
                files = sorted(per_csv_files[idx][cat])
                row.append(";".join(files))

            writer.writerow(row)

    print(f"[INFO] 汇总结果已生成：{out_path}")


if __name__ == "__main__":
    main()