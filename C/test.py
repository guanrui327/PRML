import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from math import sqrt
import matplotlib.pyplot as plt

# ================= 1. 加载训练好的最佳模型和Scaler =================
print("📦 加载模型和标准化器...")
model = load_model('best_pm25_lstm_model.h5')

# 重新拟合Scaler（使用训练集统计量）
df_train = pd.read_csv('LSTM-Multivariate_pollution.csv')
df_train['pollution'].fillna(method='ffill', inplace=True)
df_train = pd.get_dummies(df_train, columns=['wnd_dir'], prefix='wind')
feature_cols = [c for c in df_train.columns if c != 'date']
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(df_train[feature_cols])
print(f"✅ 特征维度: {len(feature_cols)}")

# ================= 2. 加载并预处理测试集 =================
print("🔍 预处理测试数据...")
df_test = pd.read_csv('pollution_test_data1.csv')

# 测试集列名处理（原始数据无列名，手动指定）
df_test.columns = ['dew', 'temp', 'press', 'wnd_dir', 'wnd_spd', 'snow', 'rain', 'pollution']

# 风向编码（必须与训练集一致！）
df_test = pd.get_dummies(df_test, columns=['wnd_dir'], prefix='wind')

# 对齐特征列（测试集可能缺少某些风向的one-hot列）
for col in feature_cols:
    if col not in df_test.columns and col != 'pollution':
        df_test[col] = 0  # 缺失的风向列补0
# 确保列顺序与训练集一致
df_test = df_test[feature_cols]

# 标准化（⚠️ 使用训练集的scaler，不能重新fit！）
test_scaled = scaler.transform(df_test)
print(f"✅ 测试集形状: {test_scaled.shape}")

# ================= 3. 构建测试序列（滑动窗口预测） =================
def prepare_test_sequences(train_scaled, test_scaled, target_idx, n_steps):
    """
    为测试集构建预测序列
    策略：用训练集末尾n_steps个样本作为初始历史，然后逐个滑动预测
    """
    X_test, y_test = [], []
    # 初始历史 = 训练集末尾
    history = train_scaled[-n_steps:].copy()
    
    for i in range(len(test_scaled)):
        # 当前输入序列
        X_test.append(history.copy())
        # 真实目标值
        y_test.append(test_scaled[i, target_idx])
        # 更新历史：移除最旧，添加新样本（用真实值更新，模拟实际部署）
        history = np.vstack([history[1:], test_scaled[i:i+1]])
    
    return np.array(X_test), np.array(y_test)

n_steps = 24  # 与训练时一致
target_idx = feature_cols.index('pollution')

# 获取训练集末尾的标准化数据用于初始化历史
train_scaled_full = scaler.transform(df_train[feature_cols])
X_test, y_test_true = prepare_test_sequences(train_scaled_full, test_scaled, target_idx, n_steps)
print(f"✅ 测试序列形状: X={X_test.shape}, y={y_test_true.shape}")

# ================= 4. 模型预测 =================
print("🚀 开始预测...")
y_pred_scaled = model.predict(X_test, verbose=0).flatten()

# ================= 5. 反标准化 & 评估 =================
def inverse_transform_single(values, scaler, target_idx, n_features):
    """对单列目标值反标准化"""
    dummy = np.zeros((len(values), n_features))
    dummy[:, target_idx] = values
    inv = scaler.inverse_transform(dummy)
    return inv[:, target_idx]

# 反标准化
y_pred = inverse_transform_single(y_pred_scaled, scaler, target_idx, len(feature_cols))
y_true = inverse_transform_single(y_test_true, scaler, target_idx, len(feature_cols))

# 计算指标
rmse = sqrt(mean_squared_error(y_true, y_pred))
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)
mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100  # 避免除0

print("\n" + "="*50)
print("📊 测试集评估结果")
print("="*50)
print(f"   RMSE:  {rmse:.2f} μg/m³")
print(f"   MAE:   {mae:.2f} μg/m³")
print(f"   R²:    {r2:.4f}")
print(f"   MAPE:  {mape:.2f}%")
print("="*50)

# ================= 6. 可视化 =================
plt.figure(figsize=(16, 5))

# 预测对比图
plt.subplot(1, 2, 1)
plt.plot(y_true[:200], label='Actual', linewidth=2, alpha=0.9)
plt.plot(y_pred[:200], label='Predicted', linewidth=2, alpha=0.9)
plt.title('PM2.5 Prediction vs Actual (First 200 samples)')
plt.xlabel('Time Step'); plt.ylabel('PM2.5 (μg/m³)')
plt.legend(); plt.grid(True, alpha=0.3)

# 误差分布图
plt.subplot(1, 2, 2)
errors = y_pred - y_true
plt.hist(errors, bins=50, edgecolor='black', alpha=0.7)
plt.axvline(0, color='red', linestyle='--', linewidth=1)
plt.title('Prediction Error Distribution')
plt.xlabel('Error (μg/m³)'); plt.ylabel('Frequency')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('test_prediction_results.png', dpi=300, bbox_inches='tight')
print("📈 结果图已保存: test_prediction_results.png")
plt.show()

# ================= 7. 保存预测结果 =================
results_df = pd.DataFrame({
    'actual_pm25': y_true,
    'predicted_pm25': y_pred,
    'error': y_pred - y_true
})
results_df.to_csv('test_prediction_results.csv', index=False)
print("💾 预测结果已保存: test_prediction_results.csv")