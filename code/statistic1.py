import pandas as pd
import numpy as np

# 读取Excel文件
df = pd.read_excel(r'D:\work\论文\Experiment\Statistics\metrics.xlsx')

# 定义统计函数
def calculate_statistics(data, column_name):
    """计算指定列的基本统计量"""
    if len(data) == 0:
        return {
            'Max': np.nan,
            'Min': np.nan,
            'Mean': np.nan,
            'Median': np.nan,
            'Std': np.nan,
            'Count': 0
        }
    return {
        'Max': data[column_name].max(),
        'Min': data[column_name].min(),
        'Mean': data[column_name].mean(),
        'Median': data[column_name].median(),
        'Std': data[column_name].std(),
        'Count': data[column_name].count()
    }

# 初始化结果字典
results = {}

# 获取所有会议
conferences = df['journal'].unique()
metrics = ['referenceCount', 'citationCount', 'influentialCitationCount']
stat_types = ['Max', 'Min', 'Total Mean', 'Median', 'Std', 'Count']

# 计算每个会议的统计
for conf in conferences:
    # 筛选当前会议的数据
    conf_data = df[df['journal'] == conf]
    
    # 分离获奖和未获奖论文
    bp_data = conf_data[conf_data['award'] == 1]  # BP: award=1
    cp_data = conf_data[conf_data['award'] == 0]  # CP: award=0
    
    conf_results = {}
    
    # 对每个指标计算统计量
    for metric in metrics:
        # 计算BP统计量
        bp_stats = calculate_statistics(bp_data, metric)
        # 计算CP统计量
        cp_stats = calculate_statistics(cp_data, metric)
        
        # 将结果存储到字典中
        conf_results[metric] = {
            'BP': bp_stats,
            'CP': cp_stats
        }
    
    results[conf] = conf_results

# 计算所有会议合并的Total统计（分BP和CP）
all_bp_data = df[df['award'] == 1]  # 所有会议的获奖论文
all_cp_data = df[df['award'] == 0]  # 所有会议的未获奖论文

all_conferences_stats = {}
for metric in metrics:
    all_conferences_stats[metric] = {
        'BP': calculate_statistics(all_bp_data, metric),
        'CP': calculate_statistics(all_cp_data, metric)
    }

# 创建最终的统计表格
final_rows = []

# 为每个会议添加行
for conf in conferences:
    # 添加会议标题行
    final_rows.append({'Conf.': conf, 'Metrics': ''})
    
    # 为每个指标添加统计行
    for stat in stat_types:
        # 将'Total Mean'映射到'Mean'进行数据提取
        stat_key = 'Mean' if stat == 'Total Mean' else stat
        
        row = {'Conf.': '', 'Metrics': stat}
        
        for metric in metrics:
            if conf in results:
                row[f'{metric}_BP'] = results[conf][metric]['BP'].get(stat_key, np.nan)
                row[f'{metric}_CP'] = results[conf][metric]['CP'].get(stat_key, np.nan)
            else:
                row[f'{metric}_BP'] = np.nan
                row[f'{metric}_CP'] = np.nan
        
        final_rows.append(row)
    
    # 添加空行分隔不同的会议
    final_rows.append({'Conf.': '', 'Metrics': ''})

# 添加所有会议汇总的Total部分
final_rows.append({'Conf.': 'Total', 'Metrics': ''})  # Total标题行

for stat in stat_types:
    stat_key = 'Mean' if stat == 'Total Mean' else stat
    row = {'Conf.': '', 'Metrics': stat}
    
    for metric in metrics:
        row[f'{metric}_BP'] = all_conferences_stats[metric]['BP'].get(stat_key, np.nan)
        row[f'{metric}_CP'] = all_conferences_stats[metric]['CP'].get(stat_key, np.nan)
    
    final_rows.append(row)

# 转换为DataFrame
final_df = pd.DataFrame(final_rows)

# 重新排序列顺序
column_order = ['Conf.', 'Metrics']
for metric in metrics:
    column_order.extend([f'{metric}_BP', f'{metric}_CP'])

final_df = final_df[column_order]

# 保存到Excel文件
output_file = 'descriptive_statistics_final.xlsx'
final_df.to_excel(output_file, index=False)

print(f"描述统计表已保存到: {output_file}")

# 显示表格预览（包含Total部分）
print("\n=== 表格预览 (包含Total部分) ===")

# 找到Total部分开始的行
total_start_idx = None
for i, row in final_df.iterrows():
    if row['Conf.'] == 'Total':
        total_start_idx = i
        break

if total_start_idx is not None:
    # 显示Total部分之前的几行和Total部分
    start_idx = max(0, total_start_idx - 10)
    end_idx = min(len(final_df), total_start_idx + 7)
    print(f"\n显示行 {start_idx+1}-{end_idx+1}:")
    print(final_df.iloc[start_idx:end_idx].to_string())
else:
    print("\n完整表格预览:")
    print(final_df.tail(20).to_string())

# 显示汇总统计摘要
print("\n=== 所有会议汇总统计摘要 ===")
print(f"总论文数: {len(df)}篇")
print(f"获奖论文数(BP): {len(all_bp_data)}篇 ({len(all_bp_data)/len(df)*100:.1f}%)")
print(f"未获奖论文数(CP): {len(all_cp_data)}篇 ({len(all_cp_data)/len(df)*100:.1f}%)")

print("\n获奖论文(BP)统计:")
for metric in metrics:
    mean_val = all_bp_data[metric].mean()
    print(f"  {metric}: 均值={mean_val:.2f}, 中位数={all_bp_data[metric].median():.2f}")

print("\n未获奖论文(CP)统计:")
for metric in metrics:
    mean_val = all_cp_data[metric].mean()
    print(f"  {metric}: 均值={mean_val:.2f}, 中位数={all_cp_data[metric].median():.2f}")

# 显示每个会议的论文数量分布
print("\n=== 各会议论文数量分布 ===")
conf_counts = df['journal'].value_counts()
for conf, count in conf_counts.items():
    award_count = len(df[(df['journal'] == conf) & (df['award'] == 1)])
    award_percent = award_count / count * 100
    print(f"{conf}: {count}篇 (获奖: {award_count}篇, {award_percent:.1f}%)")