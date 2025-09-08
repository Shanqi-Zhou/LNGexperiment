#!/usr/bin/env python3
"""
LNG实验过拟合分析脚本
深入分析模型是否存在过拟合问题
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, validation_curve, learning_curve
from sklearn.metrics import r2_score, mean_squared_error
import logging
import json
from pathlib import Path
import warnings
import sys

warnings.filterwarnings('ignore')
sys.path.insert(0, 'src')

def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def load_data():
    """加载数据"""
    logger = setup_logging()
    
    data = pd.read_csv('data/sim_lng/full_simulation_data.csv', low_memory=False)
    numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    data_clean = data[numeric_columns].copy()
    data_clean = data_clean.fillna(data_clean.median())
    
    X = data_clean.drop(columns=['orv_Q_MW'])
    y = data_clean['orv_Q_MW']
    
    return X, y

def analyze_data_characteristics(X, y):
    """分析数据特征，检查是否存在过拟合的根源"""
    logger = setup_logging()
    
    logger.info("=" * 80)
    logger.info("数据特征分析 - 过拟合风险评估")
    logger.info("=" * 80)
    
    # 1. 数据规模分析
    n_samples, n_features = X.shape
    logger.info(f"数据规模: {n_samples} 样本, {n_features} 特征")
    logger.info(f"样本/特征比: {n_samples/n_features:.1f}:1")
    
    # 2. 目标变量分析
    logger.info(f"\n目标变量统计:")
    logger.info(f"  均值: {y.mean():.6f}")
    logger.info(f"  标准差: {y.std():.6f}")
    logger.info(f"  变异系数: {(y.std()/y.mean()*100):.2f}%")
    logger.info(f"  范围: [{y.min():.6f}, {y.max():.6f}]")
    
    # 3. 检查数据重复
    duplicate_rows = X.duplicated().sum()
    logger.info(f"\n重复行数量: {duplicate_rows} ({duplicate_rows/len(X)*100:.2f}%)")
    
    # 4. 特征相关性分析
    corr_with_target = X.corrwith(y).abs().sort_values(ascending=False)
    high_corr_features = corr_with_target[corr_with_target > 0.9]
    logger.info(f"\n高相关性特征 (|r| > 0.9): {len(high_corr_features)} 个")
    
    if len(high_corr_features) > 0:
        logger.info("前5个最高相关性特征:")
        for i, (feature, corr) in enumerate(high_corr_features.head().items()):
            logger.info(f"  {i+1}. {feature}: {corr:.4f}")
    
    # 5. 检查数据泄露风险
    # 寻找与目标变量几乎完全相关的特征
    perfect_corr_features = corr_with_target[corr_with_target > 0.99]
    if len(perfect_corr_features) > 0:
        logger.info(f"\n⚠️ 警告: 发现 {len(perfect_corr_features)} 个几乎完美相关的特征 (|r| > 0.99):")
        for feature, corr in perfect_corr_features.items():
            logger.info(f"  - {feature}: {corr:.6f}")
        logger.info("  这可能表明存在数据泄露或目标变量的直接计算关系!")
    
    # 6. 特征间共线性分析
    feature_corr = X.corr().abs()
    high_inter_corr = []
    for i in range(len(feature_corr.columns)):
        for j in range(i+1, len(feature_corr.columns)):
            if feature_corr.iloc[i,j] > 0.95:
                high_inter_corr.append((feature_corr.columns[i], feature_corr.columns[j], feature_corr.iloc[i,j]))
    
    logger.info(f"\n高共线性特征对 (|r| > 0.95): {len(high_inter_corr)} 对")
    if len(high_inter_corr) > 5:
        logger.info("前5对:")
        for feat1, feat2, corr in high_inter_corr[:5]:
            logger.info(f"  {feat1} - {feat2}: {corr:.4f}")
    
    return {
        'n_samples': n_samples,
        'n_features': n_features,
        'duplicate_rows': duplicate_rows,
        'high_corr_features': len(high_corr_features),
        'perfect_corr_features': len(perfect_corr_features),
        'high_inter_corr': len(high_inter_corr),
        'target_variance': y.var(),
        'perfect_corr_list': list(perfect_corr_features.index) if len(perfect_corr_features) > 0 else []
    }

def cross_validation_analysis(X, y):
    """交叉验证分析检测过拟合"""
    logger = setup_logging()
    
    logger.info("\n" + "=" * 80)
    logger.info("交叉验证分析 - 过拟合检测")
    logger.info("=" * 80)
    
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import KFold, TimeSeriesSplit
    
    # 使用多种交叉验证策略
    cv_strategies = {
        'KFold_5': KFold(n_splits=5, shuffle=True, random_state=42),
        'KFold_10': KFold(n_splits=10, shuffle=True, random_state=42),
        'TimeSeries': TimeSeriesSplit(n_splits=5)  # 考虑时间序列特性
    }
    
    models = {
        'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        'Ridge': Ridge(alpha=1.0)
    }
    
    results = {}
    
    for model_name, model in models.items():
        logger.info(f"\n{model_name} 交叉验证结果:")
        model_results = {}
        
        for cv_name, cv_strategy in cv_strategies.items():
            try:
                scores = cross_val_score(model, X, y, cv=cv_strategy, scoring='r2')
                
                logger.info(f"  {cv_name}:")
                logger.info(f"    平均 R²: {scores.mean():.6f}")
                logger.info(f"    标准差: {scores.std():.6f}")
                logger.info(f"    范围: [{scores.min():.6f}, {scores.max():.6f}]")
                logger.info(f"    变异系数: {(scores.std()/abs(scores.mean())*100):.3f}%")
                
                model_results[cv_name] = {
                    'mean': scores.mean(),
                    'std': scores.std(),
                    'min': scores.min(),
                    'max': scores.max(),
                    'scores': scores.tolist()
                }
                
            except Exception as e:
                logger.warning(f"    {cv_name} 失败: {e}")
                model_results[cv_name] = None
        
        results[model_name] = model_results
    
    # 分析交叉验证结果
    logger.info("\n过拟合风险评估:")
    
    for model_name, model_results in results.items():
        logger.info(f"\n{model_name}:")
        for cv_name, cv_result in model_results.items():
            if cv_result is not None:
                cv_coef = cv_result['std'] / abs(cv_result['mean']) * 100
                if cv_coef < 1:
                    risk_level = "低 ✓"
                elif cv_coef < 5:
                    risk_level = "中等 ⚠️"
                else:
                    risk_level = "高 ❌"
                
                logger.info(f"  {cv_name}: 变异系数 {cv_coef:.3f}% - 过拟合风险: {risk_level}")
    
    return results

def learning_curve_analysis(X, y):
    """学习曲线分析"""
    logger = setup_logging()
    
    logger.info("\n" + "=" * 80)
    logger.info("学习曲线分析")
    logger.info("=" * 80)
    
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import learning_curve
    
    # 选择代表性样本进行学习曲线分析（大数据集太慢）
    if len(X) > 50000:
        indices = np.random.choice(len(X), 50000, replace=False)
        X_sample = X.iloc[indices]
        y_sample = y.iloc[indices]
        logger.info("使用5万样本子集进行学习曲线分析")
    else:
        X_sample = X
        y_sample = y
    
    model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
    
    train_sizes = np.linspace(0.1, 1.0, 10)
    
    try:
        train_sizes_abs, train_scores, val_scores = learning_curve(
            model, X_sample, y_sample, 
            train_sizes=train_sizes,
            cv=5,
            scoring='r2',
            n_jobs=-1,
            random_state=42
        )
        
        train_mean = train_scores.mean(axis=1)
        train_std = train_scores.std(axis=1)
        val_mean = val_scores.mean(axis=1)
        val_std = val_scores.std(axis=1)
        
        logger.info("学习曲线结果:")
        logger.info("样本量 | 训练R² | 验证R² | 差异 | 过拟合判断")
        logger.info("-" * 60)
        
        for i, size in enumerate(train_sizes_abs):
            diff = train_mean[i] - val_mean[i]
            if diff > 0.1:
                overfitting = "严重过拟合 ❌"
            elif diff > 0.05:
                overfitting = "轻微过拟合 ⚠️"
            elif diff > 0.01:
                overfitting = "可能过拟合 ?"
            else:
                overfitting = "正常 ✓"
                
            logger.info(f"{size:6d} | {train_mean[i]:.4f} | {val_mean[i]:.4f} | {diff:+.4f} | {overfitting}")
        
        # 保存学习曲线数据
        learning_curve_data = {
            'train_sizes': train_sizes_abs.tolist(),
            'train_scores_mean': train_mean.tolist(),
            'train_scores_std': train_std.tolist(),
            'val_scores_mean': val_mean.tolist(),
            'val_scores_std': val_std.tolist()
        }
        
        return learning_curve_data
        
    except Exception as e:
        logger.error(f"学习曲线分析失败: {e}")
        return None

def residual_analysis(X, y):
    """残差分析"""
    logger = setup_logging()
    
    logger.info("\n" + "=" * 80)
    logger.info("残差分析")
    logger.info("=" * 80)
    
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor
    
    # 数据分割
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 训练模型
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    # 预测
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # 计算残差
    residuals_train = y_train - y_pred_train
    residuals_test = y_test - y_pred_test
    
    logger.info("残差统计:")
    logger.info(f"训练集残差:")
    logger.info(f"  均值: {residuals_train.mean():.8f}")
    logger.info(f"  标准差: {residuals_train.std():.8f}")
    logger.info(f"  范围: [{residuals_train.min():.8f}, {residuals_train.max():.8f}]")
    
    logger.info(f"测试集残差:")
    logger.info(f"  均值: {residuals_test.mean():.8f}")
    logger.info(f"  标准差: {residuals_test.std():.8f}")
    logger.info(f"  范围: [{residuals_test.min():.8f}, {residuals_test.max():.8f}]")
    
    # 残差分析
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    
    logger.info(f"\nRMSE比较:")
    logger.info(f"  训练集 RMSE: {train_rmse:.8f}")
    logger.info(f"  测试集 RMSE: {test_rmse:.8f}")
    logger.info(f"  RMSE比值 (测试/训练): {test_rmse/train_rmse:.4f}")
    
    if test_rmse/train_rmse > 2.0:
        logger.info("  ❌ 严重过拟合: 测试集误差是训练集的2倍以上")
    elif test_rmse/train_rmse > 1.5:
        logger.info("  ⚠️ 可能过拟合: 测试集误差明显高于训练集")
    elif test_rmse/train_rmse > 1.1:
        logger.info("  ? 轻微过拟合: 测试集误差略高于训练集")
    else:
        logger.info("  ✓ 泛化性能良好: 测试集和训练集误差相当")
    
    return {
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'rmse_ratio': test_rmse/train_rmse,
        'residuals_train_stats': {
            'mean': residuals_train.mean(),
            'std': residuals_train.std(),
            'min': residuals_train.min(),
            'max': residuals_train.max()
        },
        'residuals_test_stats': {
            'mean': residuals_test.mean(),
            'std': residuals_test.std(),
            'min': residuals_test.min(),
            'max': residuals_test.max()
        }
    }

def comprehensive_overfitting_analysis():
    """综合过拟合分析"""
    logger = setup_logging()
    
    logger.info("🔍 开始LNG实验过拟合综合分析")
    logger.info("分析异常高精度指标的原因")
    
    # 加载数据
    X, y = load_data()
    
    # 1. 数据特征分析
    data_analysis = analyze_data_characteristics(X, y)
    
    # 2. 交叉验证分析
    cv_analysis = cross_validation_analysis(X, y)
    
    # 3. 学习曲线分析
    learning_curve_data = learning_curve_analysis(X, y)
    
    # 4. 残差分析
    residual_data = residual_analysis(X, y)
    
    # 综合结论
    logger.info("\n" + "=" * 80)
    logger.info("📋 综合过拟合分析结论")
    logger.info("=" * 80)
    
    # 过拟合风险评估
    risk_factors = []
    
    if data_analysis['perfect_corr_features'] > 0:
        risk_factors.append(f"存在{data_analysis['perfect_corr_features']}个几乎完美相关的特征，可能存在数据泄露")
    
    if data_analysis['duplicate_rows'] > len(X) * 0.01:
        risk_factors.append(f"重复数据占比{data_analysis['duplicate_rows']/len(X)*100:.1f}%，可能影响泛化能力")
    
    if residual_data['rmse_ratio'] > 1.5:
        risk_factors.append(f"测试集RMSE是训练集的{residual_data['rmse_ratio']:.2f}倍，存在过拟合")
    
    if len(risk_factors) == 0:
        logger.info("✅ 过拟合风险评估: 低风险")
        logger.info("   模型表现优异，泛化能力良好")
    else:
        logger.info("⚠️ 发现过拟合风险因素:")
        for i, factor in enumerate(risk_factors, 1):
            logger.info(f"   {i}. {factor}")
    
    # 保存分析结果
    analysis_results = {
        'data_characteristics': data_analysis,
        'cross_validation': cv_analysis,
        'learning_curve': learning_curve_data,
        'residual_analysis': residual_data,
        'risk_factors': risk_factors,
        'overall_risk': 'low' if len(risk_factors) == 0 else 'medium' if len(risk_factors) < 3 else 'high'
    }
    
    output_dir = Path('results')
    with open(output_dir / 'overfitting_analysis.json', 'w', encoding='utf-8') as f:
        json.dump(analysis_results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\n📁 分析结果已保存到: {output_dir}/overfitting_analysis.json")
    
    return analysis_results

if __name__ == '__main__':
    results = comprehensive_overfitting_analysis()