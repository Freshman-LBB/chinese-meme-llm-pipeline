import matplotlib.pyplot as plt
import numpy as np

# ==============================
# 1. 数据定义（单位：百分比）
# ==============================

labels = [
    "TOXICN-Random\n(Qwen-VL 7B, orig.)",
    "TOXICN-Random\n(Qwen3-vl-plus)",
    "Human-Data\n(no context)",
    "Human-Data\n(with context)",
    "LLM-Data3\n(no context)",
    "LLM-Data3\n(with context)",
]

# Precision / Recall / F1（百分数）
P = np.array([59.53, 63.52, 43.33, 77.55, 65.25, 71.11])
R = np.array([58.74, 66.67, 44.83, 49.35, 62.59, 66.67])
F1 = np.array([58.87, 65.05, 44.07, 60.32, 63.89, 68.82])

# Accuracy：前两项（TOXICN-Random）没有准确率，这里只给后四个
acc_labels = [
    "Human-Data\n(no context)",
    "Human-Data\n(with context)",
    "LLM-Data3\n(no context)",
    "LLM-Data3\n(with context)",
]
Acc = np.array([55.41, 67.53, 77.87, 81.53])

# ==============================
# 2. 画图设置
# ==============================

plt.rcParams["font.sans-serif"] = ["Arial"]  # 如果中文乱码，可以换成 SimHei 等
plt.rcParams["axes.unicode_minus"] = False

fig, axes = plt.subplots(1, 2, figsize=(14, 5))  # 左右两个子图
ax1, ax2 = axes

# -------- 左子图：P / R / F1 --------
x = np.arange(len(labels))
width = 0.25  # 每个柱的宽度

ax1.bar(x - width, P, width, label="Precision", color="#4C72B0")
ax1.bar(x,         R, width, label="Recall",    color="#55A868")
ax1.bar(x + width, F1, width, label="F1-score", color="#C44E52")

ax1.set_ylabel("Score (%)")
ax1.set_title("LLM Performance on Different Datasets\n(P / R / F1)")
ax1.set_xticks(x)
ax1.set_xticklabels(labels, rotation=20, ha="right")
ax1.set_ylim(0, 100)
ax1.legend()

# 在柱顶标注数值（可选，如果你觉得太挤可以删掉这段）
def add_labels(ax, x_pos, values):
    for x_val, y_val in zip(x_pos, values):
        ax.text(x_val, y_val + 1, f"{y_val:.1f}", ha="center", va="bottom", fontsize=8)

add_labels(ax1, x - width, P)
add_labels(ax1, x,         R)
add_labels(ax1, x + width, F1)

# -------- 右子图：Accuracy --------
x_acc = np.arange(len(acc_labels))
width_acc = 0.5

bars = ax2.bar(x_acc, Acc, width_acc, color="#8172B3", label="Accuracy")

ax2.set_ylabel("Accuracy (%)")
ax2.set_title("Accuracy on Datasets with Available Labels")
ax2.set_xticks(x_acc)
ax2.set_xticklabels(acc_labels, rotation=20, ha="right")
ax2.set_ylim(0, 100)
ax2.legend()

# 在柱顶标注 Accuracy 数值
for x_val, y_val in zip(x_acc, Acc):
    ax2.text(x_val, y_val + 1, f"{y_val:.1f}", ha="center", va="bottom", fontsize=8)

plt.tight_layout()

# ==============================
# 3. 保存与显示
# ==============================

plt.savefig("llm_performance_barchart.svg", dpi=300, bbox_inches="tight")
plt.show()