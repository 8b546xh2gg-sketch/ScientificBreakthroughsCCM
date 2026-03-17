import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
import os

# 设置英文样式
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 设置美观的样式
sns.set_style("whitegrid")
plt.style.use('seaborn-v0_8-talk')

# 从表格中提取数据
data = {
    'Citation_Type': ['Deep', 'Deep', 'Moderate', 'Moderate', 'Shallow', 'Shallow'],
    'Paper_Type': ['BP', 'CP', 'BP', 'CP', 'BP', 'CP'],
    'Max': [9, 24, 58, 86, 49, 72],
    'Min': [0, 0, 0, 0, 3, 0],
    'Mean': [1.17, 1.37, 11.76, 12.02, 18.53, 13.35],
    'Std': [1.78, 2.15, 9.51, 9.67, 9.96, 9.10],
    'Count': [272, 836, 2739, 7333, 4318, 8144],
    'Proportion': [0.04, 0.05, 0.37, 0.45, 0.59, 0.50]
}

# 创建DataFrame
df = pd.DataFrame(data)

# 设置颜色方案 - 更淡雅的颜色
bp_color = '#FFB74D'  # 浅橙色
cp_color = '#64B5F6'  # 浅蓝色

# 创建figures文件夹（如果不存在）
figures_dir = r'D:\work\论文\Experiment\Statistics\56_1'  
if not os.path.exists(figures_dir):
    os.makedirs(figures_dir)

# ==================== 创建图形 - 调整左右比例 ====================
# 减小左图比例，增大右图比例
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8),  # 增加总宽度
                                gridspec_kw={'width_ratios': [1, 1.2]})  # 左:右 = 1:1.2

# ==================== LEFT PANEL: Stacked Bar Chart ====================
# Prepare data for stacked bar chart
citation_types = df['Citation_Type'].unique()
bp_counts = df[df['Paper_Type'] == 'BP']['Count'].values
cp_counts = df[df['Paper_Type'] == 'CP']['Count'].values
bp_props = df[df['Paper_Type'] == 'BP']['Proportion'].values
cp_props = df[df['Paper_Type'] == 'CP']['Proportion'].values

# 设置更大的字体和图形参数
bar_height = 0.5
fontsize_labels = 14  # 标签字体大小
fontsize_numbers = 12  # 数字字体大小
fontsize_axis = 13    # 坐标轴字体大小

# Draw stacked horizontal bars
bars_bp = ax1.barh(citation_types, bp_counts, color=bp_color, 
                   edgecolor='white', linewidth=2, label='BP', height=bar_height)  # 修改2：图例改为BP
bars_cp = ax1.barh(citation_types, cp_counts, left=bp_counts, color=cp_color, 
                   edgecolor='white', linewidth=2, label='CP', height=bar_height)  # 修改2：图例改为CP

# 增加数字标签的字体大小
for i, (type_name, bp_count, cp_count, bp_prop, cp_prop) in enumerate(
    zip(citation_types, bp_counts, cp_counts, bp_props, cp_props)):
    
    # MODIFIED: (a)子图中，文字标签只显示百分比，大小增大1.2倍，颜色灰色不加粗
    if bp_count > 0:
        ax1.text(bp_count/2, i, f'{bp_prop:.1%}',  # 只显示百分比
                ha='center', va='center', color='#262626',  # 颜色灰色
                fontsize=fontsize_numbers * 1.3)         # 大小1.2倍，不加粗（默认normal）
    
    # MODIFIED: (a)子图中，文字标签只显示百分比，大小增大1.2倍，颜色灰色不加粗
    if cp_count > 0:
        ax1.text(bp_count + cp_count/2, i, f'{cp_prop:.1%}',  # 只显示百分比
                ha='center', va='center', color='#262626',         # 颜色灰色
                fontsize=fontsize_numbers * 1.3)                 # 大小1.2倍，不加粗

# Style the left chart
ax1.set_xlabel('Number of Papers', fontsize=fontsize_labels)
ax1.set_title('(a) Citation Volume and Composition', fontsize=16, pad=30, y=-0.2, fontweight='bold')  # 修改3：加(a)和加粗
ax1.tick_params(axis='y', labelsize=fontsize_axis)
ax1.tick_params(axis='x', labelsize=fontsize_axis-1)
ax1.invert_yaxis()  # Invert Y-axis to have Deep at the top
ax1.set_xlim(0, max(bp_counts + cp_counts) * 1.1)

# 使纵轴标签竖向排列
ax1.set_yticklabels(ax1.get_yticklabels(), rotation=90, va='center')

# Add grid lines
ax1.grid(True, axis='x', linestyle='--', alpha=0.3)

# 为左图添加图例（简化版）- 修改：位置改为右上方
ax1.legend(loc='upper right', fontsize=11, frameon=True)  # 修改2：位置从'lower right'改为'upper right'

# ==================== RIGHT PANEL: Mean and Standard Deviation Chart ====================
# 准备数据
x_pos = np.arange(len(citation_types))
bar_width = 0.35

# 获取数据
bp_means = df[df['Paper_Type'] == 'BP']['Mean'].values
cp_means = df[df['Paper_Type'] == 'CP']['Mean'].values
bp_stds = df[df['Paper_Type'] == 'BP']['Std'].values
cp_stds = df[df['Paper_Type'] == 'CP']['Std'].values
bp_max = df[df['Paper_Type'] == 'BP']['Max'].values
cp_max = df[df['Paper_Type'] == 'CP']['Max'].values

# 增加误差线和柱状图的视觉权重
error_kw_config = {'elinewidth': 2.5, 'capthick': 2.5, 'capsize': 10}

# Draw grouped bar chart with error bars
bars_bp_mean = ax2.bar(x_pos - bar_width/2, bp_means, bar_width, 
                       color=bp_color, edgecolor='white', linewidth=2.5,
                       yerr=bp_stds, error_kw=error_kw_config,
                       label='BP', alpha=0.9)  # 修改2：图例改为BP
bars_cp_mean = ax2.bar(x_pos + bar_width/2, cp_means, bar_width, 
                       color=cp_color, edgecolor='white', linewidth=2.5,
                       yerr=cp_stds, error_kw=error_kw_config,
                       label='CP', alpha=0.9)  # 修改2：图例改为CP

# MODIFIED: (b)子图中，均值±标准差标签颜色改为灰色不加粗
for i, (bp_mean, cp_mean, bp_std, cp_std) in enumerate(zip(bp_means, cp_means, bp_stds, cp_stds)):
    # BP label - 颜色灰色，不加粗
    ax2.text(i - bar_width/2, bp_mean + bp_std + 1.5, 
             f'{bp_mean:.2f} ± {bp_std:.2f}', 
             ha='center', va='bottom', fontsize=fontsize_numbers * 1.1, 
             color='#262626')  # 不加粗（默认normal）
    
    # CP label - 颜色灰色，不加粗
    ax2.text(i + bar_width/2, cp_mean + cp_std + 1.5, 
             f'{cp_mean:.2f} ± {cp_std:.2f}', 
             ha='center', va='bottom', fontsize=fontsize_numbers * 1.1, 
             color='#262626')  # 不加粗

# 增加最大值标记的大小
marker_size = 12
for i, (bp_max_val, cp_max_val) in enumerate(zip(bp_max, cp_max)):
    ax2.plot(i - bar_width/2, bp_max_val, '^', color=bp_color, 
             markersize=marker_size * 1.1, alpha=0.8, markeredgecolor='white', markeredgewidth=1.5)
    ax2.plot(i + bar_width/2, cp_max_val, '^', color=cp_color, 
             markersize=marker_size * 1.1, alpha=0.8, markeredgecolor='white', markeredgewidth=1.5)
    
    # MODIFIED: (b)子图中，最大值标签只显示值，颜色灰色不加粗
    vertical_offset = 3 if i == 0 else 4  # 根据类型调整偏移量
    if bp_max_val > bp_means[i] + bp_stds[i] + 3:
        ax2.text(i - bar_width/2, bp_max_val + vertical_offset, f'{bp_max_val}',  # 只显示值
                 ha='center', va='bottom', fontsize=fontsize_numbers * 1.1, 
                 color='#262626', alpha=0.8)  # 不加粗（默认normal）
    if cp_max_val > cp_means[i] + cp_stds[i] + 3:
        ax2.text(i + bar_width/2, cp_max_val + vertical_offset, f'{cp_max_val}',  # 只显示值
                 ha='center', va='bottom', fontsize=fontsize_numbers * 1.1, 
                 color='#262626', alpha=0.8)  # 不加粗

# Style the right chart
ax2.set_xlabel('Citation Type', fontsize=fontsize_labels)
ax2.set_ylabel('Mean Citations ± Standard Deviation', fontsize=fontsize_labels)
ax2.set_title('(b) Statistical Characteristics of Citations', fontsize=16,  pad=30, y=-0.2, fontweight='bold')  # 修改3：加(b)和加粗
ax2.set_xticks(x_pos)
ax2.set_xticklabels(citation_types, fontsize=fontsize_axis)
ax2.tick_params(axis='y', labelsize=fontsize_axis-1)

# 设置Y轴范围
max_y_val = max(np.max(bp_means + bp_stds), np.max(cp_means + cp_stds), np.max(bp_max), np.max(cp_max))
ax2.set_ylim(0, max_y_val * 1.25)  # 增加顶部空间以容纳标签

# 添加右侧图例（简化）
handles = [Patch(facecolor=bp_color, edgecolor='white', label='BP'),  # 修改2：图例改为BP
           Patch(facecolor=cp_color, edgecolor='white', label='CP'),  # 修改2：图例改为CP
           plt.Line2D([0], [0], marker='^', color='gray', markersize=10, 
                     linestyle='None', markeredgecolor='white', label='Maximum Value')]
ax2.legend(handles=handles, loc='upper left', fontsize=11, frameon=True)

# 添加网格线
ax2.grid(True, axis='y', linestyle='--', alpha=0.3)

# 调整布局，减少左图空间
plt.tight_layout()
# 修改1：左右图之间的空隙缩小为原来的二分之一（wspace从0.3改为0.15）
plt.subplots_adjust(top=0.95, bottom=0.1, wspace=0.15)  # 修改1：空隙从0.3缩小为0.15

# ==================== SAVE FIGURES ====================
# 定义文件路径
png_path = os.path.join(figures_dir, 'citation_analysis_combo_chart.png')
pdf_path = os.path.join(figures_dir, 'citation_analysis_combo_chart.pdf')
high_res_png_path = os.path.join(figures_dir, 'citation_analysis_combo_chart_highres.png')

# 保存图表
plt.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Saved chart to: {png_path}")

plt.savefig(pdf_path, bbox_inches='tight', facecolor='white')
print(f"Saved chart to: {pdf_path}")

plt.savefig(high_res_png_path, dpi=600, bbox_inches='tight', facecolor='white')
print(f"Saved high-resolution chart to: {high_res_png_path}")

# 显示图表
plt.show()

print("\n图表已生成，主要改进：")
print("1. 调整左右比例：左图占比减少，右图占比增加")
print("2. 增大所有文字标签：数字标签12pt，轴标签13-14pt")
print("3. 增加图形元素大小：误差线、标记、柱状图轮廓")
print(f"4. 所有文件保存至：{figures_dir}/")