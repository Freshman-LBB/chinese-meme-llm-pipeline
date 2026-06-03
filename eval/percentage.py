import matplotlib.pyplot as plt
import pandas as pd
import os
import sys

# 设置学术绘图风格
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 14
plt.rcParams['axes.unicode_minus'] = False


def draw_pie_chart(labels, sizes, colors, explode,output_dir, output_filename):
    """
    绘制并保存饼状图的通用函数
    """
    fig, ax = plt.subplots(figsize=(6, 6))

    # 绘制饼图
    wedges, texts, autotexts = ax.pie(
        sizes,
        explode=explode,
        labels=labels,
        colors=colors,
        autopct='%1.1f%%',
        shadow=False,
        startangle=140,
        textprops=dict(color="black")
    )

    # 优化字体样式
    plt.setp(texts, size=14, weight="bold")
    plt.setp(autotexts, size=12, weight="bold", color="white")

    ax.axis('equal')  # 保证饼图是圆的

    # 保存图片，去除多余边框
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, output_filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"图片已保存至: {save_path}")
    plt.close()


def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # ==========================================
    # 图表 1: TOXICN 原始数据分布
    # ==========================================
    print("正在生成图表 1 (TOXICN 原始数据)...")

    # 数据来源：表格最后一行 Total
    # N-Harm: 8,173
    # Harm: 3,827
    original_labels = ['Non-Harmful', 'Harmful']
    original_sizes = [8173, 3827]
    # 学术常用配色：蓝色代表无害，红色代表有害
    original_colors = ['#5B9BD5', '#ED7D31']
    original_explode = (0, 0.05)  # 稍微分离有害部分以突出显示

    # ==========================================
    # 图表 2: 标注后数据集分布 (动态/一般)
    # ==========================================
    print("正在读取数据并生成图表 2 (标注数据集)...")

    # 构建 CSV 路径: ../../construct/dataset/feature/pick_and_divide_summary.csv
    csv_relative_path = os.path.join(
        current_dir,
        "../../construct/dataset/feature/pick_and_divide_summary.csv"
    )
    csv_path = os.path.abspath(csv_relative_path)
    output_dir = os.path.dirname(csv_path)

    draw_pie_chart(
        original_labels,
        original_sizes,
        original_colors,
        original_explode,
        output_dir,
        "toxicn_original_distribution.svg"
    )

    if not os.path.exists(csv_path):
        print(f"错误: 找不到 CSV 文件: {csv_path}")
        return

    try:
        df = pd.read_csv(csv_path)

        # 提取数据函数
        def get_count(category_name):
            # 假设列名为 '类别' 和 '总数（不去重）'
            row = df[df['类别'] == category_name]
            if row.empty:
                print(f"警告: 未在 CSV 中找到类别 '{category_name}'，默认为 0")
                return 0
            return row['总数（不去重）'].values[0]

        # 1. 读取原始数值
        raw_dynamic_harm = get_count('动态有害')
        raw_strict_harm = get_count('严格有害')
        raw_strict_non_harm = get_count('严格无害')
        raw_undetermined = get_count('无法判断')

        # 2. 执行计算逻辑
        # 逻辑：一般有害 = 严格有害 + 无法判断 + 27 + 28 + 30
        general_harmful_count = raw_strict_harm + raw_undetermined + 27 + 28 + 30

        # 逻辑：一般无害 = 严格无害
        general_non_harmful_count = raw_strict_non_harm

        # 逻辑：动态有害 = 动态有害
        dynamic_harmful_count = raw_dynamic_harm

        # 3. 准备绘图数据
        # 顺序：一般无害, 一般有害, 动态有害
        dataset_labels = ['General Non-Harmful', 'General Harmful', 'Dynamic Harmful']
        dataset_sizes = [
            general_non_harmful_count,
            general_harmful_count,
            dynamic_harmful_count
        ]

        # 配色方案：
        # 一般无害: 蓝色 (#5B9BD5)
        # 一般有害: 橙色 (#ED7D31)
        # 动态有害: 紫色 (#A5A5A5 或 #7030A0) - 突出显示
        dataset_colors = ['#5B9BD5', '#ED7D31', '#7030A0']
        dataset_explode = (0, 0, 0.08)  # 突出显示动态有害

        # 打印计算结果供核对
        print("-" * 30)
        print("图表 2 数据核对:")
        print(f"一般无害 (General Non-Harmful): {general_non_harmful_count}")
        print(
            f"一般有害 (General Harmful)    : {general_harmful_count} (严格有害{raw_strict_harm} + 无法判断{raw_undetermined} + 补85)")
        print(f"动态有害 (Dynamic Harmful)    : {dynamic_harmful_count}")
        print("-" * 30)

        draw_pie_chart(
            dataset_labels,
            dataset_sizes,
            dataset_colors,
            dataset_explode,
            output_dir,
            "constructed_dataset_distribution.svg"
        )

    except Exception as e:
        print(f"处理 CSV 或绘图时出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()