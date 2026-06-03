import matplotlib.pyplot as plt

# 如果你想统一英文字体，可以打开下面两行（可选）
# plt.rcParams["font.sans-serif"] = ["Arial"]
# plt.rcParams["axes.unicode_minus"] = False

# 1. 列名（适当加了换行，方便在图里排版）
columns = [
    "Dataset",
    "Construction Method",
    "Context",
    "Memes\n(Instances)",
    "Contextual\nHarmful Memes",
    "Generally\nHarmful Memes",
    "Generally\nHarmless Memes",
    "Initial\nHuman Agreement",
    "LLM-Dynamic\nHarmfulness Agreement",
    "Context\nExpressiveness",
]

# 2. 表格数据（与你前面描述的一致）
table_data = [
    [
        "TOXICN-MM",
        "Web crawling\n(original authors)",
        "No",
        "12000",
        "N/A",
        "3827",
        "8173",
        "0.62",
        "N/A",
        "N/A",
    ],
    [
        "TOXICN-Random",
        "Random sampling\nfrom TOXICN-MM",
        "No",
        "213",
        "N/A",
        "159",
        "54",
        "0.62",
        "N/A",
        "N/A",
    ],
    [
        "Human-Data",
        "Human-crafted\ncontextual memes",
        "Yes",
        "77 (154)",
        "77",
        "0",
        "0",
        "N/A",
        "N/A",
        "Moderate",
    ],
    [
        "LLM-Data1",
        "LLM-based generation\n(mode 1)",
        "Yes",
        "137 (274)",
        "81",
        "44",
        "12",
        "0.54",
        "1.00",
        "Over-expressive",
    ],
    [
        "LLM-Data2",
        "LLM-based generation\n(mode 2)",
        "Yes",
        "253 (506)",
        "106",
        "48",
        "99",
        "0.54",
        "1.00",
        "Over-expressive",
    ],
    [
        "LLM-Data3",
        "LLM-based generation\n(mode 2, variant)",
        "Yes",
        "266 (532)",
        "133",
        "37",
        "96",
        "0.71",
        "1.00",
        "Moderate",
    ],
]

# 3. 画图并生成表格
fig, ax = plt.subplots(figsize=(16, 4))  # 宽一些，表格更像论文里的风格
ax.axis("off")  # 不要坐标轴

table = ax.table(
    cellText=table_data,
    colLabels=columns,
    loc="center",
    cellLoc="center",   # 文本居中
    colLoc="center",    # 列标题居中
)

# 4. 样式微调
table.auto_set_font_size(False)
table.set_fontsize(9)      # 整体字体大小
table.scale(1, 1.6)        # 行高放大一点，更易读

# 设置表头背景为浅灰，并加粗
for (row, col), cell in table.get_celld().items():
    if row == 0:  # 表头行
        cell.set_facecolor("#DDDDDD")
        cell.get_text().set_weight("bold")
    # 可以根据需要调节边框粗细
    cell.set_linewidth(0.5)

# 如果 matplotlib 版本支持，可自动根据内容调整列宽
try:
    table.auto_set_column_width(col=list(range(len(columns))))
except Exception:
    pass

# 5. 布局 & 保存图片
plt.tight_layout()

# 保存为高清 PNG（适合截图/插图）
fig.savefig("dataset_table.svg", dpi=300, bbox_inches="tight")

# 如果你想直接在窗口中预览，也可以用：
plt.show()