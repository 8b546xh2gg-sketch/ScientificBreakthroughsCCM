import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# ======================
# 1. Load data
# ======================
file_path = r"D:\work\论文\Experiment\Statistics\final data.xlsx"
df = pd.read_excel(file_path)

# Map award to BP / CP
df['award_label'] = df['award'].map({0: 'CP', 1: 'BP'})

# ======================
# 2. Output directory
# ======================
output_dir = r"D:\work\论文\Experiment\Statistics\56_2"
os.makedirs(output_dir, exist_ok=True)

# ======================
# 3. Plot settings
# ======================
sns.set(style="whitegrid", font_scale=1.1)

metrics = [
    ('min_similarity', 'Minimum similarity'),
    ('mean_similarity', 'Mean similarity'),
    ('max_similarity', 'Maximum similarity')
]

# 更新颜色映射：BP用#FC8D62，CP用#8DA0CB
palette = {'BP': '#FC8D62', 'CP': '#8DA0CB'}

fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=False)

# 子图标题
subplot_titles = ['(a)', '(b)', '(c)']

# 颜色设置
box_color = '#6B6B6B'  # 统计量的颜色

for idx, (ax, (metric, ylabel), title) in enumerate(zip(axes, metrics, subplot_titles)):
    # 绘制小提琴图 - 只需要一半的小提琴图，不需要箱线图
    violin = sns.violinplot(
        data=df,
        x='award_label',
        y=metric,
        hue='award_label',
        split=True,
        inner=None,  # 不显示内部图形，我们将手动添加统计量
        cut=0,
        linewidth=1,
        palette=palette,
        ax=ax,
        legend=False
    )
    
    # 移除原有的x轴刻度标签，重新设置
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['BP', 'CP'])
    
    # 为每个组在小提琴图中添加统计量
    for i, award_type in enumerate(['BP', 'CP']):
        data = df[df['award_label'] == award_type][metric].dropna()
        
        # 计算统计量
        median_val = np.median(data)
        mean_val = np.mean(data)
        
        # 计算四分位数
        q1, q3 = np.percentile(data, [25, 75])
        
        # 确定位置
        # 在小提琴图中，由于split=True，CP在x=0的位置（右半边），BP在x=1的位置（左半边）
        if award_type == 'BP':
            # BP：在小提琴图的左半边，位置在x=1
            x_pos = 1
            # 在左半边绘制统计量，稍微靠右一点
            median_x_start = x_pos - 0.02
            median_x_end = x_pos - 0.1
            mean_x_pos = x_pos - 0.06
        else:
            # CP：在小提琴图的右半边，位置在x=0
            x_pos = 0
            # 在右半边绘制统计量，稍微靠左一点
            median_x_start = x_pos + 0.02
            median_x_end = x_pos + 0.1
            mean_x_pos = x_pos + 0.06
        
        # 在小提琴图中绘制中位数线
        ax.hlines(median_val, median_x_start, median_x_end, 
                 color=box_color, linewidth=2, zorder=5)
        
        # 在小提琴图中绘制均值点
        ax.plot(mean_x_pos, mean_val, 'o', color='white', 
                markeredgecolor=box_color, markersize=8, zorder=6)
        
        # 可选：也可以添加四分位数线
        # ax.hlines(q1, x_pos-0.05, x_pos, color=box_color, linewidth=1, linestyle='--', alpha=0.7, zorder=4)
        # ax.hlines(q3, x_pos-0.05, x_pos, color=box_color, linewidth=1, linestyle='--', alpha=0.7, zorder=4)
    
    ax.set_xlabel('')
    ax.set_ylabel(ylabel, fontsize=12)
    
    # 添加子图标题（在下方）
    ax.text(0.5, -0.15, title, transform=ax.transAxes, 
            fontsize=14, fontweight='bold', ha='center', va='center')

# 调整子图间距
plt.tight_layout(rect=[0, 0.1, 1, 0.9])

# ======================
# 4. Legend handling - 添加颜色和中位数/均值的说明
# ======================
# 创建颜色图例项
from matplotlib.patches import Patch
color_legend_elements = [
    Patch(facecolor=palette['BP'], edgecolor='black', label='BP'),
    Patch(facecolor=palette['CP'], edgecolor='black', label='CP')
]

# 创建统计量图例项
import matplotlib.lines as mlines
stat_legend_elements = [
    mlines.Line2D([], [], color=box_color, linewidth=2, linestyle='-', label='Median'),
    mlines.Line2D([], [], color='white', marker='o', markersize=8, 
                  markeredgecolor=box_color, linestyle='None', label='Mean')
]

# 合并所有图例项
all_legend_elements = color_legend_elements + stat_legend_elements

# 创建统一的图例
fig.legend(handles=all_legend_elements, 
           loc='upper center', 
           ncol=4,
           frameon=True,
           fancybox=True,
           shadow=True,
           borderpad=1,
           fontsize=11)

# ======================
# 5. Save figure
# ======================
# PNG
png_path = os.path.join(output_dir, "split_violin_statistics_BP_CP.png")
plt.savefig(png_path, dpi=300, bbox_inches='tight')

# PDF
pdf_path = os.path.join(output_dir, "split_violin_statistics_BP_CP.pdf")
plt.savefig(pdf_path, format='pdf', bbox_inches='tight')

plt.show()
plt.close()