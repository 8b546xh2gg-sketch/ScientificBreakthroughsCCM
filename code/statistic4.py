import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# ==================== 设置样式 ====================
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

# ==================== 读取数据 ====================
df = pd.read_excel(r'D:\work\论文\Experiment\Statistics\final data.xlsx')

# 按原始idx排序
df = df.sort_values('idx').reset_index(drop=True)

# 分离BP和CP数据
bp_data = df[df['award'] == 1]
cp_data = df[df['award'] == 0]

# 临界阈值
critical_threshold = 0.5704

print("数据统计摘要:")
print(f"总样本量: {len(df)}")
print(f"BP 样本量: {len(bp_data)}")
print(f"CP 样本量: {len(cp_data)}")
print(f"临界阈值 Δ* = {critical_threshold}")

# ==================== 颜色定义 ====================
bp_color = '#ff973b'  # 橙色
cp_color = '#79adcd'  # 蓝色
threshold_color = '#FF6347'  # 红色用于阈值线
mutation_region_color = '#FFE5CC'  # 浅橙色阴影用于突变区
grid_color = '#E0E0E0'  # 网格线颜色

# ==================== 创建组合图 ====================
# 修改这里：交换子图顺序，ax1现在是右图，ax2现在是左图
fig, (ax2, ax1) = plt.subplots(1, 2, figsize=(16, 7), 
                                gridspec_kw={'width_ratios': [1, 1]})  # 修改为1:1比例

# ==================== 左图（原右图）：m-n二维分布 ====================
# 准备数据（保持原始顺序）
bp_n = bp_data['n'].values
bp_m = bp_data['m'].values
cp_n = cp_data['n'].values
cp_m = cp_data['m'].values

# 绘制散点图 - 移除边缘黑线
bp_scatter_2d = ax2.scatter(bp_n, bp_m, color=bp_color, s=40, 
                           alpha=0.7, edgecolor='none', linewidth=0,
                           label='BP', zorder=3)

cp_scatter_2d = ax2.scatter(cp_n, cp_m, color=cp_color, s=40, 
                           alpha=0.7, edgecolor='none', linewidth=0,
                           label='CP', zorder=3)

# 设置左图（m-n二维分布）
ax2.set_xlabel('knowledge Relevance (n)', fontsize=12)
ax2.set_ylabel('Knowledge Heterogeneity (m)', fontsize=12)
ax2.set_title('(a) Distribution of m and n', fontsize=16, pad=30, y=-0.2, fontweight='bold')

# 设置坐标轴范围
n_min = min(bp_n.min(), cp_n.min())
n_max = max(bp_n.max(), cp_n.max())
m_min = min(bp_m.min(), cp_m.min())
m_max = max(bp_m.max(), cp_m.max())

# 增加一些边距
n_range = n_max - n_min
m_range = m_max - m_min
ax2.set_xlim(n_min - 0.05 * n_range, n_max + 0.05 * n_range)
ax2.set_ylim(m_min - 0.05 * m_range, m_max + 0.05 * m_range)

# 添加网格
ax2.grid(True, linestyle='--', alpha=0.3, color=grid_color, zorder=1)

# 添加左图图例
legend2 = ax2.legend(loc='lower right', fontsize=10, frameon=True, 
                    framealpha=0.95, edgecolor='#333333')
legend2.get_frame().set_linewidth(1.0)

# ==================== 右图（原左图）：判别式Δ分布（按原始idx排序） ====================
# 获取所有数据的Δ值，按原始idx排序
all_delta = df['discriminant'].values
all_indices = df['idx'].values  # 原始idx
award_status = df['award'].values  # 0=CP, 1=BP

# 绘制散点图（按原始idx顺序）- 移除边缘黑线
for i in range(len(df)):
    color = bp_color if award_status[i] == 1 else cp_color
    ax1.scatter(all_indices[i], all_delta[i], color=color, s=40, 
               alpha=0.8, edgecolor='none', linewidth=0, zorder=3)  # 移除边缘线

# 添加临界阈值线
ax1.axhline(y=critical_threshold, color=threshold_color, linestyle='--', 
           linewidth=2.5, zorder=2)

# 添加突变区阴影 (<Δ*)
ymin, ymax = ax1.get_ylim()
ax1.axhspan(ymin=ymin, ymax=critical_threshold, 
           facecolor=mutation_region_color, alpha=0.3, zorder=1)

# 设置右图（判别式Δ分布）
ax1.set_xlabel('Paper Index (Original Order)', fontsize=12)
ax1.set_ylabel('Discriminant Δ', fontsize=12)
ax1.set_title('(b) Distribution of the Discriminant Δ', fontsize=16, pad=30, y=-0.2, fontweight='bold')
ax1.set_xlim(all_indices.min() - 1, all_indices.max() + 1)
ax1.set_ylim(min(all_delta.min() - 0.1, ymin), max(all_delta.max() + 0.1, ymax))

# 添加网格
ax1.grid(True, axis='y', linestyle='--', alpha=0.3, color=grid_color, zorder=1)
ax1.grid(True, axis='x', linestyle=':', alpha=0.2, color=grid_color, zorder=1)

# ==================== 调整两个图的大小和间距 ====================
# 确保两个图的高度和宽度一致
# 获取两个图的坐标轴限制并统一
all_delta_range = all_delta.max() - all_delta.min()
all_n_range = n_max - n_min
all_m_range = m_max - m_min

# 计算合适的宽高比
delta_aspect = (all_indices.max() - all_indices.min()) / all_delta_range
mn_aspect = all_n_range / all_m_range

# 设置统一的图形大小
plt.tight_layout()
plt.subplots_adjust(wspace=0.3)  # 调整两个图之间的间距

# ==================== 保存图像 ====================
figures_dir = r'D:\work\论文\Experiment\Statistics\56_3'
if not os.path.exists(figures_dir):
    os.makedirs(figures_dir)

png_path = os.path.join(figures_dir, 'delta_mn_combo_chart.png')
pdf_path = os.path.join(figures_dir, 'delta_mn_combo_chart.pdf')
svg_path = os.path.join(figures_dir, 'delta_mn_combo_chart.svg')

plt.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"\n保存PNG到: {png_path}")

plt.savefig(pdf_path, bbox_inches='tight', facecolor='white')
print(f"保存PDF到: {pdf_path}")

plt.savefig(svg_path, bbox_inches='tight', facecolor='white')
print(f"保存SVG到: {svg_path}")

# ==================== 显示图表 ====================
plt.show()

# ==================== 打印详细统计（终端显示） ====================
print("\n" + "="*70)
print("详细统计分析:")
print("="*70)

# 计算统计数据
bp_delta = bp_data['discriminant'].values
cp_delta = cp_data['discriminant'].values
bp_below_threshold = np.sum(bp_delta < critical_threshold)
cp_below_threshold = np.sum(cp_delta < critical_threshold)
bp_percent = bp_below_threshold / len(bp_delta) * 100
cp_percent = cp_below_threshold / len(cp_delta) * 100

print(f"\n判别式Δ统计 (临界阈值 Δ* = {critical_threshold}):")
print(f"BP Δ < Δ*: {bp_below_threshold}/{len(bp_delta)} ({bp_percent:.1f}%)")
print(f"CP Δ < Δ*: {cp_below_threshold}/{len(cp_delta)} ({cp_percent:.1f}%)")
print(f"BP Δ范围: [{bp_delta.min():.4f}, {bp_delta.max():.4f}]")
print(f"CP Δ范围: [{cp_delta.min():.4f}, {cp_delta.max():.4f}]")
print(f"BP Δ均值: {bp_delta.mean():.4f} ± {bp_delta.std():.4f}")
print(f"CP Δ均值: {cp_delta.mean():.4f} ± {cp_delta.std():.4f}")

print("\n知识关联性(n)统计:")
print(f"BP n: 均值={bp_n.mean():.4f} ± {bp_n.std():.4f}")
print(f"范围: [{bp_n.min():.4f}, {bp_n.max():.4f}]")
print(f"CP n: 均值={cp_n.mean():.4f} ± {cp_n.std():.4f}")
print(f"范围: [{cp_n.min():.4f}, {cp_n.max():.4f}]")

print("\n知识异质性(m)统计:")
print(f"BP m: 均值={bp_m.mean():.4f} ± {bp_m.std():.4f}")
print(f"范围: [{bp_m.min():.4f}, {bp_m.max():.4f}]")
print(f"CP m: 均值={cp_m.mean():.4f} ± {cp_m.std():.4f}")
print(f"范围: [{cp_m.min():.4f}, {cp_m.max():.4f}]")

# 添加图形尺寸信息
print(f"\n图形尺寸: {fig.get_size_inches()[0]:.1f} × {fig.get_size_inches()[1]:.1f} 英寸")
print(f"点大小: 40 (无边缘线)")
print(f"颜色: BP={bp_color}, CP={cp_color}")
print("="*70)