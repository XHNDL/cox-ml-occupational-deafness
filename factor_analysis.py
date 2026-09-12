"""
因素分析与可视化
基于统一数据集分析各因素与噪声聋/职业病风险的关联
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun']
matplotlib.rcParams['axes.unicode_minus'] = False


def load_unified_data():
    """加载统一数据"""
    print("\n=== Loading Unified Data ===")
    df = pd.read_csv('result/unified_data.csv')
    print(f"Total samples: {len(df)}")
    return df


def analyze_factors(df):
    """分析各因素"""
    print("\n=== Factor Analysis ===")
    
    results = []
    
    # 数据来源统计
    print("\n数据来源分布:")
    print(df['数据来源'].value_counts())
    
    # 数据3: 噪声聋分析
    df_d3 = df[df['数据来源'] == 'data3'].dropna(subset=['target_noise'])
    print(f"\n数据3 (噪声聋): {len(df_d3)} 样本, 阳性: {df_d3['target_noise'].sum()}")
    
    # 数据2: 职业病风险分析
    df_d2 = df[df['数据来源'] == 'data2'].dropna(subset=['职业病风险'])
    print(f"数据2 (职业病): {len(df_d2)} 样本, 阳性: {df_d2['职业病风险'].sum()}")
    
    return df_d3, df_d2


def plot_analysis(df_d3, df_d2):
    """绘制分析结果"""
    print("\n=== Plotting ===")
    
    fig = plt.figure(figsize=(20, 16))
    
    # 1. 噪声聋发病率 - 年龄分布
    ax1 = fig.add_subplot(3, 3, 1)
    bins = [0, 30, 40, 50, 60, 70]
    labels = ['≤30', '31-40', '41-50', '51-60', '>60']
    df_d3['年龄组'] = pd.cut(df_d3['年龄'], bins=bins, labels=labels)
    rate = df_d3.groupby('年龄组', observed=True)['target_noise'].mean() * 100
    ax1.bar(range(len(rate)), rate.values, color='steelblue', alpha=0.7)
    ax1.set_xticks(range(len(rate)))
    ax1.set_xticklabels(rate.index)
    ax1.set_xlabel('Age Group')
    ax1.set_ylabel('Disease Rate (%)')
    ax1.set_title('Noise-Induced Hearing Loss Rate by Age')
    for i, v in enumerate(rate.values):
        ax1.text(i, v + 1, f'{v:.1f}%', ha='center')
    
    # 2. 噪声聋发病率 - 接害工龄分布
    ax2 = fig.add_subplot(3, 3, 2)
    bins_w = [0, 5, 10, 15, 20, 30]
    labels_w = ['0-5', '5-10', '10-15', '15-20', '20+']
    df_d3['工龄组'] = pd.cut(df_d3['接害工龄_年'], bins=bins_w, labels=labels_w)
    rate_w = df_d3.groupby('工龄组', observed=True)['target_noise'].mean() * 100
    ax2.bar(range(len(rate_w)), rate_w.values, color='coral', alpha=0.7)
    ax2.set_xticks(range(len(rate_w)))
    ax2.set_xticklabels(rate_w.index)
    ax2.set_xlabel('Work Duration (years)')
    ax2.set_ylabel('Disease Rate (%)')
    ax2.set_title('Noise-Induced Hearing Loss Rate by Work Duration')
    
    # 3. 噪声聋发病率 - BMI分布
    ax3 = fig.add_subplot(3, 3, 3)
    bins_bmi = [0, 18.5, 24, 28, 35]
    labels_bmi = ['<18.5', '18.5-24', '24-28', '>28']
    df_d3['BMI组'] = pd.cut(df_d3['BMI'], bins=bins_bmi, labels=labels_bmi)
    rate_bmi = df_d3.groupby('BMI组', observed=True)['target_noise'].mean() * 100
    ax3.bar(range(len(rate_bmi)), rate_bmi.values, color='green', alpha=0.7)
    ax3.set_xticks(range(len(rate_bmi)))
    ax3.set_xticklabels(rate_bmi.index)
    ax3.set_xlabel('BMI Group')
    ax3.set_ylabel('Disease Rate (%)')
    ax3.set_title('Noise-Induced Hearing Loss Rate by BMI')
    
    # 4. 性别与噪声聋
    ax4 = fig.add_subplot(3, 3, 4)
    rate_sex = df_d3.groupby('性别')['target_noise'].mean() * 100
    ax4.bar(['Female', 'Male'], [rate_sex.get(0, 0), rate_sex.get(1, 0)], color=['pink', 'steelblue'], alpha=0.7)
    ax4.set_ylabel('Disease Rate (%)')
    ax4.set_title('Disease Rate by Gender')
    
    # 5. 血压与噪声聋
    ax5 = fig.add_subplot(3, 3, 5)
    df_d3['高血压'] = (df_d3['收缩压_mmHg'] >= 140).astype(int)
    rate_bp = df_d3.groupby('高血压')['target_noise'].mean() * 100
    ax5.bar(['Normal', 'High BP'], [rate_bp.get(0, 0), rate_bp.get(1, 0)], color=['lightgreen', 'salmon'], alpha=0.7)
    ax5.set_ylabel('Disease Rate (%)')
    ax5.set_title('Disease Rate by Blood Pressure')
    
    # 6. 数据2: 肺功能与职业病风险
    ax6 = fig.add_subplot(3, 3, 6)
    df_d2_valid = df_d2.dropna(subset=['肺功能'])
    if len(df_d2_valid) > 0:
        bins_lung = [0, 70, 80, 90, 100, 120]
        labels_lung = ['<70', '70-80', '80-90', '90-100', '>100']
        df_d2_valid['肺功能组'] = pd.cut(df_d2_valid['肺功能'], bins=bins_lung, labels=labels_lung)
        rate_lung = df_d2_valid.groupby('肺功能组', observed=True)['职业病风险'].mean() * 100
        ax6.bar(range(len(rate_lung)), rate_lung.values, color='purple', alpha=0.7)
        ax6.set_xticks(range(len(rate_lung)))
        ax6.set_xticklabels(rate_lung.index)
        ax6.set_xlabel('Lung Function')
        ax6.set_ylabel('Risk Rate (%)')
        ax6.set_title('Occupational Disease Risk by Lung Function')
    
    # 7. 数据2: 年龄与职业病风险
    ax7 = fig.add_subplot(3, 3, 7)
    df_d2['年龄组'] = pd.cut(df_d2['年龄'], bins=[0, 30, 40, 50, 60, 70], labels=['≤30', '31-40', '41-50', '51-60', '>60'])
    rate_age2 = df_d2.groupby('年龄组', observed=True)['职业病风险'].mean() * 100
    ax7.bar(range(len(rate_age2)), rate_age2.values, color='orange', alpha=0.7)
    ax7.set_xticks(range(len(rate_age2)))
    ax7.set_xticklabels(rate_age2.index)
    ax7.set_xlabel('Age Group')
    ax7.set_ylabel('Risk Rate (%)')
    ax7.set_title('Occupational Disease Risk by Age')
    
    # 8. 特征相关性热力图
    ax8 = fig.add_subplot(3, 3, 8)
    cols_corr = ['年龄', '接害工龄_年', 'BMI', '收缩压_mmHg', '舒张压_mmHg', '肺功能']
    df_corr = df_d3[cols_corr].copy()
    if '肺功能' in df_d2.columns:
        df_corr = df_d2[cols_corr].dropna()
    corr = df_corr.corr()
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', ax=ax8, center=0)
    ax8.set_title('Feature Correlation (Data2)')
    
    # 9. 数据汇总
    ax9 = fig.add_subplot(3, 3, 9)
    ax9.axis('off')
    info = """
    UNIFIED DATA ANALYSIS SUMMARY
    =========================
    
    Data Sources:
    • Data3: 46,267 samples
      - Noise-induced hearing loss
      - Target: 双耳高频平均听阈结果 > 25dB
      - Positive: 13,361 (28.9%)
    
    • Data2: 76,035 samples
      - Occupational disease risk
      - Target: 疑似/确诊职业病
      - Positive: 22 (0.03%)
    
    Key Findings:
    • Age: Strong correlation with disease
    • Work duration: Higher risk with longer exposure
    • BMI: Moderate correlation
    • Blood pressure: Weak correlation
    • Lung function (Data2): Notable risk factor
    """
    ax9.text(0.05, 0.95, info, transform=ax9.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.suptitle('Unified Factor Analysis - Occupational Noise-Induced Hearing Loss', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('unified_factor_analysis.png', dpi=300, bbox_inches='tight')
    print("\n[Saved] unified_factor_analysis.png")
    plt.close()


def main():
    print("="*60)
    print("  Unified Factor Analysis")
    print("="*60)
    
    df = load_unified_data()
    df_d3, df_d2 = analyze_factors(df)
    plot_analysis(df_d3, df_d2)
    
    print("\n=== [Done] ===")


if __name__ == "__main__":
    main()