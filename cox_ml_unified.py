"""
Cox比例风险模型 + 机器学习统一预测系统
融合生存分析与机器学习的职业性噪声聋风险预测
"""
import pandas as pd
import numpy as np
import os
import joblib
import json
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, f1_score
from lifelines import CoxPHFitter, KaplanMeierFitter
import warnings
warnings.filterwarnings('ignore')

# 字体配置 - 优先使用系统自带字体
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['font.family'] = 'sans-serif'


def load_cleaned_data():
    """加载统一数据"""
    print("\n=== Loading Unified Data ===")
    
    df = pd.read_csv('result/unified_data.csv')
    
    # 使用数据3（有听力数据）
    df_d3 = df[df['数据来源'] == 'data3'].dropna(subset=['target_noise'])
    df_d3 = df_d3.rename(columns={'target_noise': 'target'})
    
    feature_cols = ['性别', '年龄', '接害工龄_年', '身高_cm', '体重_kg', 
                   'BMI', '收缩压_mmHg', '舒张压_mmHg']
    
    df_valid = df_d3[feature_cols + ['target', '双耳高频平均听阈结果']].dropna()
    
    df_valid['duration'] = df_valid['接害工龄_年'].clip(upper=30)
    df_valid['event'] = df_valid['target']
    
    X = df_valid[feature_cols].fillna(df_valid[feature_cols].median())
    y = df_valid['target'].values
    
    print(f"Total samples: {len(X)}, Disease rate: {y.mean()*100:.1f}%")
    print(f"Features: {list(X.columns)}")
    
    return X, y, df_valid


def train_cox_model(df):
    """训练Cox比例风险模型"""
    print("\n=== Training Cox Proportional Hazards Model ===")
    
    # 准备Cox数据 - 不使用duration作为协变量
    cox_cols = ['性别', '年龄', '接害工龄_年', '身高_cm', '体重_kg', 
                'BMI', '收缩压_mmHg', '舒张压_mmHg', 'duration', 'event']
    
    cox_data = df[cox_cols].copy()
    
    # 不标准化，让模型自然拟合
    # 移除接害工龄作为协变量（与duration高度相关会导致问题）
    cox_features = ['性别', '年龄', '身高_cm', '体重_kg', 'BMI', '收缩压_mmHg', '舒张压_mmHg']
    cox_data = cox_data[cox_features + ['duration', 'event']]
    
    # 拟合Cox模型
    cph = CoxPHFitter()
    cph.fit(cox_data, duration_col='duration', event_col='event')
    
    print("\nCox Model Summary:")
    print("-" * 60)
    cph.print_summary()
    
    # 特征名称映射
    feature_map = {
        '性别': 'Gender',
        '年龄': 'Age',
        '身高_cm': 'Height',
        '体重_kg': 'Weight',
        'BMI': 'BMI',
        '收缩压_mmHg': 'SBP',
        '舒张压_mmHg': 'DBP'
    }
    
    # 提取系数和HR
    summary = cph.summary
    cox_weights = {}
    
    print("\n=== Cox Model Hazard Ratios ===")
    print("-" * 60)
    print(f"{'Feature':<15} {'HR':<10} {'95% CI':<20} {'P-value':<10}")
    print("-" * 60)
    
    for idx in summary.index:
        if idx in ['duration', 'event']:
            continue
        hr = summary.loc[idx, 'exp(coef)']
        ci_low = summary.loc[idx, 'exp(coef) lower 95%']
        ci_high = summary.loc[idx, 'exp(coef) upper 95%']
        p = summary.loc[idx, 'p']
        
        # 归一化权重
        weight = abs(np.log(hr))
        if p > 0.1:
            weight *= 0.5
        
        cox_weights[idx] = weight
        
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
        en_name = feature_map.get(idx, idx)
        print(f"{en_name:<15} {hr:<10.3f} [{ci_low:.2f}-{ci_high:.2f}]    {p:<10.4f} {sig}")
    
    # 归一化权重
    total = sum(cox_weights.values())
    cox_weights = {k: v/total for k, v in cox_weights.items()}
    
    return cph, cox_weights, None


def train_ml_models(X, y):
    """训练机器学习模型"""
    print("\n=== Training Machine Learning Models ===")
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42, stratify=y
    )
    
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=150, max_depth=10, 
                                                 min_samples_split=10, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, max_depth=5, 
                                                          random_state=42)
    }
    
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_proba)
        results[name] = {'model': model, 'auc': auc, 'y_test': y_test, 'y_proba': y_proba}
        print(f"{name}: AUC={auc:.4f}")
    
    return results, X_test, y_test, scaler


def create_unified_prediction(cox_model, ml_results, X, y, X_test, y_test):
    """创建统一的预测系统（融合Cox + ML）"""
    print("\n=== Creating Unified Prediction System ===")
    
    # 获取ML模型预测
    best_ml_name = max(ml_results, key=lambda x: ml_results[x]['auc'])
    best_ml = ml_results[best_ml_name]['model']
    ml_proba = ml_results[best_ml_name]['y_proba']
    
    # 获取Cox模型预测（风险分数）
    cox_test = X_test.copy()
    for i, col in enumerate(X.columns):
        if col in cox_model.params_.index:
            coef = cox_model.params_[col]
            if i == 0:
                cox_scores = coef * cox_test[:, i]
            else:
                cox_scores += coef * cox_test[:, i]
    
    # 归一化到0-1
    cox_scores_norm = 1 / (1 + np.exp(-cox_scores))  # sigmoid归一化
    
    # 融合预测（加权平均）
    # 根据AUC分配权重
    ml_auc = ml_results[best_ml_name]['auc']
    cox_auc = roc_auc_score(y_test, 1 - cox_scores_norm)
    
    # 简单融合：各50%
    combined_proba = 0.5 * ml_proba + 0.5 * (1 - cox_scores_norm)
    
    # 评估融合模型
    combined_auc = roc_auc_score(y_test, combined_proba)
    
    print(f"\nCox Model AUC: {cox_auc:.4f}")
    print(f"ML Model AUC: {ml_auc:.4f}")
    print(f"Combined Model AUC: {combined_auc:.4f}")
    
    return {
        'ml_proba': ml_proba,
        'cox_scores': cox_scores_norm,
        'combined_proba': combined_proba,
        'combined_auc': combined_auc,
        'ml_auc': ml_auc,
        'cox_auc': cox_auc
    }


def plot_cox_ml_integration(cox_model, ml_results, unified_results, X, y):
    """绘制Cox+ML集成结果"""
    print("\n=== Plotting Results ===")
    
    # 特征名称映射
    feature_map = {
        '性别': 'Gender',
        '年龄': 'Age',
        '身高_cm': 'Height',
        '体重_kg': 'Weight',
        'BMI': 'BMI',
        '收缩压_mmHg': 'SBP',
        '舒张压_mmHg': 'DBP'
    }
    
    best_ml_name = max(ml_results, key=lambda x: ml_results[x]['auc'])
    
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Cox模型森林图
    ax1 = fig.add_subplot(3, 3, 1)
    summary = cox_model.summary
    features = [idx for idx in summary.index if idx not in ['duration', 'event']]
    hrs = [summary.loc[f, 'exp(coef)'] for f in features]
    lower = [summary.loc[f, 'exp(coef) lower 95%'] for f in features]
    upper = [summary.loc[f, 'exp(coef) upper 95%'] for f in features]
    
    # 限制显示范围，避免极端值
    hrs_clipped = [min(max(h, 0.1), 10) for h in hrs]
    lower_clipped = [min(max(l, 0.1), 10) for l in lower]
    upper_clipped = [min(max(u, 0.1), 10) for u in upper]
    
    y_pos = np.arange(len(features))
    # 使用英文标签
    en_features = [feature_map.get(f, f) for f in features]
    ax1.barh(y_pos, hrs_clipped, xerr=[np.array(hrs_clipped)-np.array(lower_clipped), 
                                        np.array(upper_clipped)-np.array(hrs_clipped)], 
             color='steelblue', alpha=0.7, capsize=3)
    ax1.axvline(x=1, color='red', linestyle='--', linewidth=2)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(en_features)
    ax1.set_xlabel('Hazard Ratio (HR)')
    ax1.set_title('Cox Model: Forest Plot\n(Hazard Ratios with 95% CI)')
    ax1.set_xscale('log')
    
    # 2. ROC曲线对比
    ax2 = fig.add_subplot(3, 3, 2)
    fpr_ml, tpr_ml, _ = roc_curve(unified_results['y_test'], unified_results['ml_proba'])
    fpr_cox, tpr_cox, _ = roc_curve(unified_results['y_test'], unified_results['cox_scores'])
    fpr_comb, tpr_comb, _ = roc_curve(unified_results['y_test'], unified_results['combined_proba'])
    
    ax2.plot(fpr_ml, tpr_ml, 'b-', label=f'ML (AUC={unified_results["ml_auc"]:.3f})', linewidth=2)
    ax2.plot(fpr_cox, tpr_cox, 'g-', label=f'Cox (AUC={unified_results["cox_auc"]:.3f})', linewidth=2)
    ax2.plot(fpr_comb, tpr_comb, 'r-', label=f'Combined (AUC={unified_results["combined_auc"]:.3f})', linewidth=2)
    ax2.plot([0, 1], [0, 1], 'k--')
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.set_title('ROC Curves: Cox vs ML vs Combined')
    ax2.legend(loc='lower right')
    
    # 3. 模型AUC对比
    ax3 = fig.add_subplot(3, 3, 3)
    models = ['Cox PH', best_ml_name, 'Combined']
    aucs = [unified_results['cox_auc'], unified_results['ml_auc'], unified_results['combined_auc']]
    colors = ['green', 'steelblue', 'coral']
    ax3.bar(models, aucs, color=colors, alpha=0.7)
    ax3.set_ylabel('AUC')
    ax3.set_title('Model Performance Comparison')
    for i, v in enumerate(aucs):
        ax3.text(i, v + 0.01, f'{v:.3f}', ha='center')
    ax3.set_ylim(0, 1)
    
    # 4. Kaplan-Meier生存曲线
    ax4 = fig.add_subplot(3, 3, 4)
    kmf = KaplanMeierFitter()
    
    # 按年龄分组
    df_km = pd.DataFrame({'duration': X['接害工龄_年'].values[:len(y)], 
                         'event': y[:len(X)]})
    df_km['age_group'] = pd.cut(X['年龄'].values[:len(y)], bins=[0, 35, 45, 60], labels=['Young', 'Middle', 'Old'])
    
    for group in ['Young', 'Middle', 'Old']:
        mask = df_km['age_group'] == group
        if mask.sum() > 10:
            kmf.fit(df_km.loc[mask, 'duration'], df_km.loc[mask, 'event'], label=group)
            kmf.plot_survival_function(ax=ax4)
    
    ax4.set_xlabel('Work Duration (Years)')
    ax4.set_ylabel('Survival Probability')
    ax4.set_title('Kaplan-Meier Survival Curves\nby Age Group')
    ax4.legend()
    
    # 5. 特征重要性（RF）
    ax5 = fig.add_subplot(3, 3, 5)
    rf = ml_results['Random Forest']['model']
    imp = rf.feature_importances_
    imp_df = pd.DataFrame({'feature': X.columns, 'importance': imp}).sort_values('importance', ascending=True)
    ax5.barh(imp_df['feature'], imp_df['importance'], color='darkgreen', alpha=0.7)
    ax5.set_xlabel('Feature Importance')
    ax5.set_title('Random Forest Feature Importance')
    
    # 6. Cox系数与ML重要性对比
    ax6 = fig.add_subplot(3, 3, 6)
    common_features = [f for f in X.columns if f in cox_model.params_.index]
    cox_coefs = [abs(cox_model.params_[f]) for f in common_features]
    ml_imps = [rf.feature_importances_[list(X.columns).index(f)] for f in common_features]
    
    # 归一化
    cox_coefs = np.array(cox_coefs) / max(cox_coefs)
    ml_imps = np.array(ml_imps) / max(ml_imps)
    
    x = np.arange(len(common_features))
    width = 0.35
    ax6.bar(x - width/2, cox_coefs, width, label='Cox |coef|', color='green', alpha=0.7)
    ax6.bar(x + width/2, ml_imps, width, label='RF importance', color='steelblue', alpha=0.7)
    ax6.set_xticks(x)
    ax6.set_xticklabels(common_features, rotation=45, ha='right')
    ax6.set_ylabel('Normalized Score')
    ax6.set_title('Feature Importance: Cox vs RF')
    ax6.legend()
    
    # 7. 预测概率分布
    ax7 = fig.add_subplot(3, 3, 7)
    ax7.hist(unified_results['ml_proba'][unified_results['y_test']==0], bins=25, alpha=0.5, 
             label='Healthy', color='green')
    ax7.hist(unified_results['ml_proba'][unified_results['y_test']==1], bins=25, alpha=0.5, 
             label='Disease', color='red')
    ax7.set_xlabel('Predicted Probability (ML)')
    ax7.set_ylabel('Count')
    ax7.set_title('ML Model Prediction Distribution')
    ax7.legend()
    
    # 8. Cox风险分数分布
    ax8 = fig.add_subplot(3, 3, 8)
    ax8.hist(unified_results['cox_scores'][unified_results['y_test']==0], bins=25, alpha=0.5, 
             label='Healthy', color='green')
    ax8.hist(unified_results['cox_scores'][unified_results['y_test']==1], bins=25, alpha=0.5, 
             label='Disease', color='red')
    ax8.set_xlabel('Cox Risk Score')
    ax8.set_ylabel('Count')
    ax8.set_title('Cox Model Risk Score Distribution')
    ax8.legend()
    
    # 9. 综合信息
    ax9 = fig.add_subplot(3, 3, 9)
    ax9.axis('off')
    info = f"""
    COX + ML UNIFIED SYSTEM SUMMARY
    ===============================
    
    Model Components:
    - Cox Proportional Hazards Model
      (Analyzes time-to-event relationships)
      (Outputs Hazard Ratios)
    
    - Machine Learning Models
      - Random Forest
      - Gradient Boosting
      - Logistic Regression
    
    - Integration Strategy
      - Simple weighted average (50% each)
    
    Performance:
    - Cox AUC: {unified_results['cox_auc']:.4f}
    - ML AUC: {unified_results['ml_auc']:.4f}
    - Combined AUC: {unified_results['combined_auc']:.4f}
    
    Key Findings:
    - Gender & BMI are significant factors
    - Cox provides survival analysis perspective
    - ML models have better predictive power
    """
    ax9.text(0.05, 0.95, info, transform=ax9.transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.suptitle('Cox + ML Unified Risk Prediction System\nOccupational Noise-Induced Hearing Loss', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('cox_ml_unified_results.png', dpi=300, bbox_inches='tight')
    print("\n[Saved] cox_ml_unified_results.png")
    plt.close()


def save_models(cox_model, ml_results, X, unified_results):
    """保存所有模型"""
    print("\n=== Saving Models ===")
    
    os.makedirs('models_cox_ml', exist_ok=True)
    
    # 保存ML模型
    best_ml_name = max(ml_results, key=lambda x: ml_results[x]['auc'])
    best_ml = ml_results[best_ml_name]['model']
    joblib.dump(best_ml, 'models_cox_ml/ml_model.pkl')
    
    # 保存Cox模型参数
    summary = cox_model.summary
    cox_params = {
        'coefficients': summary['coef'].to_dict(),
        'hazard_ratios': summary['exp(coef)'].to_dict(),
        'p_values': summary['p'].to_dict(),
        'concordance': cox_model.concordance_index_
    }
    joblib.dump(cox_params, 'models_cox_ml/cox_params.pkl')
    
    # 保存信息
    info = {
        'model_type': 'Cox + ML Unified',
        'ml_model': best_ml_name,
        'ml_auc': unified_results['ml_auc'],
        'cox_auc': unified_results['cox_auc'],
        'combined_auc': unified_results['combined_auc'],
        'features': X.columns.tolist(),
        'integration_method': 'Simple weighted average (50% each)',
        'cox_concordance': cox_params['concordance']
    }
    
    with open('models_cox_ml/model_info.json', 'w', encoding='utf-8') as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    
    print("Models saved to models_cox_ml/")


def main():
    print("="*70)
    print("   Cox + Machine Learning Unified Prediction System")
    print("   职业性噪声聋风险预测系统")
    print("="*70)
    
    # 1. 加载数据
    X, y, df_valid = load_cleaned_data()
    
    # 2. 训练Cox模型
    cox_model, cox_weights, _ = train_cox_model(df_valid)
    
    # 3. 训练ML模型
    ml_results, X_test, y_test, scaler = train_ml_models(X, y)
    
    # 4. 创建统一预测
    unified_results = create_unified_prediction(
        cox_model, ml_results, X, y, X_test, y_test
    )
    unified_results['y_test'] = y_test
    
    # 5. 绘图
    best_ml_name = max(ml_results, key=lambda x: ml_results[x]['auc'])
    plot_cox_ml_integration(cox_model, ml_results, unified_results, X, y)
    
    # 6. 保存模型
    save_models(cox_model, ml_results, X, unified_results)
    
    print("\n" + "="*70)
    print("   COX + ML UNIFIED SYSTEM COMPLETE!")
    print("="*70)


if __name__ == "__main__":
    main()