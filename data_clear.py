"""
数据加载与清洗模块
用途：将原始体检数据（数据2+数据3）转换为统一格式，用于职业性噪声聋风险预测
"""
import pandas as pd
import numpy as np
import os
import re


def load_data3():
    """加载数据3（有听力数据 - 噪声聋）"""
    print("\n=== [加载] 数据3（有听力）.xlsx ===")
    project_root = os.getcwd()
    input_file = os.path.join(project_root, 'data', '数据3（有听力）.xlsx')
    
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"数据文件不存在: {input_file}")
    
    df = pd.read_excel(input_file, header=1)
    print(f"原始数据形状: {df.shape}")
    return df


def load_data2():
    """加载数据2（体检数据 - 职业病关联分析）"""
    print("\n=== [加载] 数据2.xlsx ===")
    project_root = os.getcwd()
    input_file = os.path.join(project_root, 'data', '数据2.xlsx')
    
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"数据文件不存在: {input_file}")
    
    df = pd.read_excel(input_file, header=1)
    print(f"原始数据形状: {df.shape}")
    return df


def clean_data3(df):
    """清洗数据3"""
    print("\n--- [清洗] 数据3 ---")
    
    # 通过位置索引提取各列
    df_clean = pd.DataFrame({
        '性别': df.iloc[:, 2].map({'男': 1, '女': 0}),
        '年龄': pd.to_numeric(df.iloc[:, 3], errors='coerce'),
        '接害工龄_年': df.iloc[:, 4].apply(convert_work_years),
        '身高_cm': pd.to_numeric(df.iloc[:, 5], errors='coerce'),
        '体重_kg': pd.to_numeric(df.iloc[:, 9], errors='coerce'),
        'BMI': pd.to_numeric(df.iloc[:, 11], errors='coerce'),
        '收缩压_mmHg': pd.to_numeric(df.iloc[:, 14], errors='coerce'),
        '舒张压_mmHg': pd.to_numeric(df.iloc[:, 17], errors='coerce'),
        '双耳高频平均听阈结果': pd.to_numeric(df.iloc[:, 24], errors='coerce')
    })
    
    df_clean['数据来源'] = 'data3'
    
    # 异常值清洗
    clean_outliers(df_clean)
    
    print(f"数据3清洗后: {len(df_clean)} 行")
    return df_clean


def clean_data2(df):
    """清洗数据2"""
    print("\n--- [清洗] 数据2 ---")
    
    data = pd.DataFrame()
    
    # 基础信息
    data['性别'] = df.iloc[:, 2].map({'男': 1, '女': 0})
    data['年龄'] = pd.to_numeric(df.iloc[:, 3], errors='coerce')
    data['接害工龄_年'] = df.iloc[:, 5].apply(convert_work_years)
    data['身高_cm'] = pd.to_numeric(df.iloc[:, 6], errors='coerce')
    data['体重_kg'] = pd.to_numeric(df.iloc[:, 10], errors='coerce')
    data['BMI'] = pd.to_numeric(df.iloc[:, 12], errors='coerce')
    data['收缩压_mmHg'] = pd.to_numeric(df.iloc[:, 14], errors='coerce')
    data['舒张压_mmHg'] = pd.to_numeric(df.iloc[:, 17], errors='coerce')
    
    # 肺功能
    data['肺功能'] = pd.to_numeric(df.iloc[:, 23], errors='coerce')
    
    # 心电图、胸片
    data['心电图异常'] = (df.iloc[:, 22].fillna('正常') != '正常').astype(int)
    data['胸片异常'] = (df.iloc[:, 27].fillna('正常') != '正常').astype(int)
    
    # 职业病风险标签
    disease = df.iloc[:, 29].fillna('')
    data['职业病风险'] = disease.str.contains('疑似职业病|职业病确诊', regex=True).astype(int)
    
    data['数据来源'] = 'data2'
    
    # 异常值清洗
    clean_outliers(data)
    
    print(f"数据2清洗后: {len(data)} 行")
    return data


def convert_work_years(s):
    """接害工龄格式转换"""
    if pd.isna(s):
        return np.nan
    try:
        s = str(s)
        y = re.search(r'(\d+)年', s)
        m = re.search(r'(\d+)月', s)
        years = int(y.group(1)) if y else 0
        months = int(m.group(1)) if m else 0
        return years + months / 12
    except:
        return np.nan


def clean_outliers(df):
    """异常值清洗"""
    # 收缩压: 合理范围 60-200
    if '收缩压_mmHg' in df.columns:
        df.loc[(df['收缩压_mmHg'] < 60) | (df['收缩压_mmHg'] > 200), '收缩压_mmHg'] = np.nan
    # 舒张压: 合理范围 40-130
    if '舒张压_mmHg' in df.columns:
        df.loc[(df['舒张压_mmHg'] < 40) | (df['舒张压_mmHg'] > 130), '舒张压_mmHg'] = np.nan
    # BMI: 合理范围 10-60
    if 'BMI' in df.columns:
        df.loc[(df['BMI'] < 10) | (df['BMI'] > 60), 'BMI'] = np.nan
    # 身高: 合理范围 100-220
    if '身高_cm' in df.columns:
        df.loc[(df['身高_cm'] < 100) | (df['身高_cm'] > 220), '身高_cm'] = np.nan
    # 体重: 合理范围 30-200
    if '体重_kg' in df.columns:
        df.loc[(df['体重_kg'] < 30) | (df['体重_kg'] > 200), '体重_kg'] = np.nan
    # 听力阈值: 合理范围 -10 ~ 100
    if '双耳高频平均听阈结果' in df.columns:
        df.loc[(df['双耳高频平均听阈结果'] < -10) | (df['双耳高频平均听阈结果'] > 100), '双耳高频平均听阈结果'] = np.nan
    # 肺功能
    if '肺功能' in df.columns:
        df.loc[(df['肺功能'] < 30) | (df['肺功能'] > 150), '肺功能'] = np.nan
    # 去除幻影值
    df = df.replace([999, 999.0, 999.9, np.inf], np.nan)
    return df


def merge_datasets(df3, df2):
    """合并两个数据集"""
    print("\n--- [合并] 数据2 + 数据3 ---")
    
    # 数据3: 有听力阈值 -> 可计算目标变量（噪声聋）
    df3['target_noise'] = (df3['双耳高频平均听阈结果'] > 25).astype(int)
    
    # 数据2: 已有职业病风险标签
    # 填充数据2中缺失的列（与数据3对齐）
    for col in ['双耳高频平均听阈结果', 'target_noise']:
        if col not in df2.columns:
            df2[col] = np.nan
    
    # 统一列顺序
    common_cols = ['性别', '年龄', '接害工龄_年', '身高_cm', '体重_kg', 'BMI', 
                   '收缩压_mmHg', '舒张压_mmHg', '肺功能', '心电图异常', '胸片异常',
                   '双耳高频平均听阈结果', 'target_noise', '职业病风险', '数据来源']
    
    df3_subset = df3[[c for c in common_cols if c in df3.columns]]
    df2_subset = df2[[c for c in common_cols if c in df2.columns]]
    
    df_merged = pd.concat([df3_subset, df2_subset], ignore_index=True)
    
    # 填充缺失值
    for col in df_merged.columns:
        if col not in ['target_noise', '职业病风险', '数据来源']:
            df_merged[col] = df_merged[col].fillna(df_merged[col].median())
    
    print(f"合并后: {len(df_merged)} 行")
    print(f"  - 数据3: {len(df3)} 行 (噪声聋目标: {df3['target_noise'].sum()})")
    print(f"  - 数据2: {len(df2)} 行 (职业病风险: {df2['职业病风险'].sum()})")
    
    return df_merged


def save_data(df, output_path='result/unified_data.csv'):
    """保存清洗后的数据"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n数据已保存至: {output_path}")


def main():
    print("="*60)
    print("  数据加载与清洗 - 统一数据集")
    print("="*60)
    
    # 加载数据
    df3_raw = load_data3()
    df2_raw = load_data2()
    
    # 清洗数据
    df3_clean = clean_data3(df3_raw)
    df2_clean = clean_data2(df2_raw)
    
    # 合并数据
    df_unified = merge_datasets(df3_clean, df2_clean)
    
    # 统计
    print("\n=== [统一数据集统计] ===")
    print(f"总样本数: {len(df_unified)}")
    print(f"\n各列缺失值:")
    print(df_unified.isnull().sum())
    print(f"\n数值列统计:")
    print(df_unified.describe())
    
    # 保存
    save_data(df_unified)
    
    print("\n=== [完成] ===")


if __name__ == "__main__":
    main()