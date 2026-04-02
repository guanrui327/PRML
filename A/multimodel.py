# -*- coding: utf-8 -*-
# 高斯过程回归拟合 - Data4Regression.xlsx

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, ExpSineSquared, WhiteKernel, DotProduct
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ============ 1. 数据加载 ============
def load_data(filepath='Data4Regression.xlsx'):
    train = pd.read_excel(filepath, sheet_name=0)
    test = pd.read_excel(filepath, sheet_name=1)
    train.columns = train.columns.str.strip()
    test.columns = test.columns.str.strip()
    
    X_train = train['x'].values.reshape(-1, 1)
    y_train = train['y_complex'].values
    X_test = test['x_new'].values.reshape(-1, 1)
    y_test = test['y_new_complex'].values
    return X_train, y_train, X_test, y_test

# ============ 2. 高斯过程回归建模 ============
def build_gp_model(X_train, y_train, X_test, y_test, kernel_config='periodic'):
    """构建高斯过程回归模型，返回完整评估指标"""
    
    if kernel_config == 'periodic':
        # 复合核：周期性 + 局部平滑 + 趋势 + 噪声
        kernel = (
            C(1.0, (1e-3, 1e3)) * 
            ExpSineSquared(length_scale=1.0, periodicity=2.8, 
                          length_scale_bounds=(1e-2, 1e2),
                          periodicity_bounds=(1.0, 10.0)) +
            C(0.5, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2)) +
            C(0.1, (1e-3, 1e3)) * DotProduct(sigma_0=1.0) +
            WhiteKernel(0.1, (1e-5, 1e1))
        )
        kernel_name = "Periodic+RBF+Trend+Noise"
    else:
        kernel = C(1.0) * RBF(1.0)
        kernel_name = "RBF-Baseline"
    
    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=0.01,
        normalize_y=True,
        n_restarts_optimizer=10,
        random_state=42
    )
    
    print(f"🔧 核函数: {kernel_name}")
    print("🔄 优化核超参数...")
    gp.fit(X_train, y_train)
    
    # 预测
    y_train_pred = gp.predict(X_train)
    y_test_pred, y_test_std = gp.predict(X_test, return_std=True)
    
    # ✅ 修复：确保返回所有需要的指标
    return {
        'model': gp,
        'kernel': gp.kernel_,
        'kernel_name': kernel_name,
        'train_r2': r2_score(y_train, y_train_pred),
        'train_mse': mean_squared_error(y_train, y_train_pred),  # 新增
        'test_r2': r2_score(y_test, y_test_pred),
        'test_mse': mean_squared_error(y_test, y_test_pred),
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)), # 新增（绘图需要）
        'test_mae': mean_absolute_error(y_test, y_test_pred),
        'y_test_pred': y_test_pred,
        'y_test_std': y_test_std,
        'log_marginal_likelihood': gp.log_marginal_likelihood()
    }

# ============ 3. 核参数分析（递归修复版） ============
def analyze_kernel(kernel, indent=0):
    """递归解析复合核参数"""
    prefix = "  " * indent
    if hasattr(kernel, 'k1') and hasattr(kernel, 'k2'):
        analyze_kernel(kernel.k1, indent + 1)
        analyze_kernel(kernel.k2, indent + 1)
    else:
        kernel_type = type(kernel).__name__
        if 'ExpSineSquared' in kernel_type:
            print(f"{prefix}• 周期性核: T={kernel.periodicity:.3f}, ℓ={kernel.length_scale:.3f}")
        elif 'RBF' in kernel_type:
            print(f"{prefix}• RBF核: ℓ={kernel.length_scale:.3f}")
        elif 'WhiteKernel' in kernel_type:
            print(f"{prefix}• 噪声核: σ²={kernel.noise_level:.4f}")
        elif 'DotProduct' in kernel_type:
            print(f"{prefix}• 趋势核: σ₀={kernel.sigma_0:.3f}")
        elif 'ConstantKernel' in kernel_type:
            print(f"{prefix}• 常数核: C={kernel.constant_value:.3f}")

# ============ 4. 可视化 ============
def plot_gp_results(X_train, y_train, X_test, y_test, results):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    x_plot = np.linspace(0, 10, 500).reshape(-1, 1)
    
    # 图1: 拟合曲线 + 置信区间
    ax = axes[0, 0]
    y_plot, y_std = results['model'].predict(x_plot, return_std=True)
    ax.scatter(X_train, y_train, alpha=0.5, label='训练数据', color='blue', s=20)
    ax.scatter(X_test, y_test, alpha=0.5, label='测试数据', color='orange', s=20)
    ax.plot(x_plot, y_plot, 'r-', linewidth=2, label='GPR预测')
    ax.fill_between(x_plot.ravel(), y_plot - 2*y_std, y_plot + 2*y_std, alpha=0.2, color='red', label='95%置信区间')
    ax.set_xlabel('x'); ax.set_ylabel('y')
    ax.set_title(f'高斯过程回归拟合\n{results["kernel_name"]}\n测试R²={results["test_r2"]:.4f}')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    
    # 图2: 预测值 vs 真实值
    ax = axes[0, 1]
    ax.scatter(y_test, results['y_test_pred'], alpha=0.7, edgecolors='k', s=30)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='理想预测')
    ax.set_xlabel('真实值'); ax.set_ylabel('预测值')
    ax.set_title(f'测试集预测性能\nR²={results["test_r2"]:.4f}, RMSE={results["test_rmse"]:.4f}')
    ax.legend(); ax.grid(True, alpha=0.3)
    
    # 图3: 残差分析
    ax = axes[1, 0]
    residuals = y_test - results['y_test_pred']
    ax.scatter(results['y_test_pred'], residuals, alpha=0.7, edgecolors='k', s=30)
    ax.axhline(0, color='red', linestyle='--', linewidth=1)
    ax.set_xlabel('预测值'); ax.set_ylabel('残差')
    ax.set_title(f'残差分布\n均值={np.mean(residuals):.4f}, 标准差={np.std(residuals):.4f}')
    ax.grid(True, alpha=0.3)
    
    # 图4: 不确定性分析
    ax = axes[1, 1]
    ax.scatter(results['y_test_pred'], results['y_test_std'], alpha=0.7, edgecolors='k', s=30)
    ax.set_xlabel('预测值'); ax.set_ylabel('预测标准差σ')
    ax.set_title(f'预测不确定性\n平均σ={np.mean(results["y_test_std"]):.4f}')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('gp_regression_results.png', dpi=300, bbox_inches='tight')
    print("📈 结果图已保存: gp_regression_results.png")
    plt.show()

# ============ 5. 主流程 ============
def main(filepath='Data4Regression.xlsx'):
    print("="*70)
    print("🔬 高斯过程回归拟合 - Data4Regression.xlsx (最终修复版)")
    print("="*70)
    
    X_train, y_train, X_test, y_test = load_data(filepath)
    print(f"\n📊 数据概览: 训练{len(X_train)}样本, 测试{len(X_test)}样本")
    
    print(f"\n🚀 训练模型...")
    results = build_gp_model(X_train, y_train, X_test, y_test, kernel_config='periodic')
    
    print(f"\n📊 实验结果:")
    print(f"   🔹 训练集: R²={results['train_r2']:.4f}, MSE={results['train_mse']:.4f}")
    print(f"   🔹 测试集: R²={results['test_r2']:.4f}, MSE={results['test_mse']:.4f}, MAE={results['test_mae']:.4f}")
    
    print(f"\n🔍 核参数分析:")
    analyze_kernel(results['kernel'])
    
    residuals = y_test - results['y_test_pred']
    print(f"\n🔍 残差诊断:")
    print(f"   • 均值: {np.mean(residuals):.4f}, 标准差: {np.std(residuals):.4f}")
    print(f"   • 自相关(滞后1): {np.corrcoef(residuals[:-1], residuals[1:])[0,1]:.4f}")
    
    plot_gp_results(X_train, y_train, X_test, y_test, results)
    return results

if __name__ == "__main__":
    try:
        main()
        print("\n🎉 流程执行完毕！")
    except Exception as e:
        print(f"❌ 运行错误: {e}")
        import traceback
        traceback.print_exc()