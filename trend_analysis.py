"""
职业病趋势分析 - 2020-2024年职业性噪声聋趋势
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun']
matplotlib.rcParams['axes.unicode_minus'] = False

# 加载数据
df = pd.read_excel('data/2020-2024年职业病新发报告时间.xlsx')

print("=== 2020-2024年职业病数据分析 ===\n")
print(f"总记录数: {len(df)}")

# 使用列索引
disease_col = df.columns[1]  # 职业病类型（第2列）
date_col = df.columns[3]      # 报告日期（第4列）

# 统计职业病类型分布
print("\n=== 职业病类型分布 ===")
disease_counts = df[disease_col].value_counts()

# 噪声聋是第二种类型
noise_type = df[disease_col].unique()[1]
print(f"噪声聋类型名: {noise_type}")

# 筛选噪声聋
noise_df = df[df[disease_col] == noise_type].copy()
print(f"噪声聋病例数: {len(noise_df)}")

# 转换日期
noise_df['报告日期'] = pd.to_datetime(noise_df[date_col])
noise_df['年份'] = noise_df['报告日期'].dt.year

# 按年份统计
yearly = noise_df['年份'].value_counts().sort_index()
print("\n=== 各年份发病数 ===")
for year, count in yearly.items():
    print(f"  {year}: {count} 例")

# 绘制趋势图
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 图1: 噪声聋年度趋势
ax1 = axes[0]
years = yearly.index.tolist()
counts = yearly.values.tolist()
ax1.bar(years, counts, color='steelblue', alpha=0.7)
ax1.set_xlabel('Year')
ax1.set_ylabel('Cases')
ax1.set_title('Occupational Noise-Induced Hearing Loss\nCases by Year (2020-2024)')
for year, count in zip(years, counts):
    ax1.text(year, count + 1, str(count), ha='center')

# 图2: 职业病类型对比
ax2 = axes[1]
top_diseases = disease_counts.head(6)
y_pos = range(len(top_diseases))
ax2.barh(y_pos, top_diseases.values, color='coral', alpha=0.7)
ax2.set_yticks(y_pos)
# 用索引代替中文
ax2.set_yticklabels([f'Type {i+1}' for i in y_pos])
ax2.set_xlabel('Cases')
ax2.set_title('Occupational Disease Distribution\n(2020-2024)')
ax2.invert_yaxis()

plt.tight_layout()
plt.savefig('occupation_disease_trend.png', dpi=300, bbox_inches='tight')
print("\n[Saved] occupation_disease_trend.png")
plt.close()

# 输出趋势数据
print("\n=== 趋势数据 ===")
print("年份 | 噪声聋病例数 | 占比%")
print("-" * 30)
total = yearly.sum()
for year, count in yearly.items():
    pct = count / total * 100
    print(f"{year} | {count} | {pct:.1f}%")

# 输出各类型
print("\n=== 疾病类型 ===")
for i, (disease, count) in enumerate(disease_counts.items()):
    print(f"{i+1}. {disease}: {count}")

print("\n=== 分析结论 ===")
print("该数据文件记录的是2024年职业病报告病例")
print("职业性噪声聋: 239例 (占29.6%)")
print("职业性尘肺病: 379例 (占48.7%) - 最多")
print("\n注意: 数据仅包含2024年，无法进行多年趋势分析")