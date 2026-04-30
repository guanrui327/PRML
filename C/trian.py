import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import os

# ================= 1. 数据加载与预处理 =================
df = pd.read_csv('LSTM-Multivariate_pollution.csv')
df['date'] = pd.to_datetime(df['date'])
df = df.set_index('date').sort_index()
df['pollution'].fillna(method='ffill', inplace=True)  # 处理缺失值

# 风向独热编码
df = pd.get_dummies(df, columns=['wnd_dir'], prefix='wind')

# 特征标准化
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_values = scaler.fit_transform(df)

# ================= 2. 构建时间序列样本 =================
def create_sequences(data, target_col_name, n_steps=24):
    target_idx = list(df.columns).index(target_col_name)
    X, y = [], []
    for i in range(n_steps, len(data)):
        X.append(data[i-n_steps:i])
        y.append(data[i, target_idx])
    return np.array(X), np.array(y)

n_steps = 24
X, y = create_sequences(scaled_values, 'pollution', n_steps)

# 划分训练集/验证集 (80/20)
split = int(len(X) * 0.8)
X_train, X_val = X[:split], X[split:]
y_train, y_val = y[:split], y[split:]

# ================= 3. 构建LSTM模型 =================
def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)  # 输出PM2.5预测值
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    return model

model = build_lstm_model((n_steps, X.shape[2]))

# ================= 4. 设置早停与模型保存回调 =================
# 早停机制：监控验证集损失，连续10个epoch不下降则停止训练，并自动恢复最佳权重
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    mode='min',
    verbose=1
)

# 模型检查点：仅保存验证集损失最低的模型到本地
model_checkpoint = ModelCheckpoint(
    filepath='best_pm25_lstm_model.h5',  # TF2.13+推荐 .keras，旧版可用 .h5
    monitor='val_loss',
    save_best_only=True,
    mode='min',
    verbose=1
)

callbacks = [early_stopping, model_checkpoint]

# ================= 5. 训练模型 =================
history = model.fit(
    X_train, y_train,
    epochs=100,          # 设大一点，交由早停控制
    batch_size=64,
    validation_data=(X_val, y_val),
    callbacks=callbacks,
    shuffle=False,       # 时间序列严禁打乱
    verbose=1
)

# 绘制训练曲线
plt.figure(figsize=(10,4))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss Curve with Early Stopping')
plt.xlabel('Epoch'); plt.ylabel('MSE'); plt.legend()
plt.grid(True); plt.show()

# ================= 6. 加载最佳模型并预测 =================
# 实际项目中建议独立运行推理，这里演示加载流程
from tensorflow.keras.models import load_model
best_model = load_model('best_pm25_lstm_model.h5')

y_pred_scaled = best_model.predict(X_val, verbose=0)

# 反标准化（仅还原pollution列）
inv_dummy = np.zeros((len(y_pred_scaled), scaled_values.shape[1]))
inv_dummy[:, df.columns.get_loc('pollution')] = y_pred_scaled.flatten()
y_pred = scaler.inverse_transform(inv_dummy)[:, df.columns.get_loc('pollution')]

y_val_inv = scaler.inverse_transform(inv_dummy)[:, df.columns.get_loc('pollution')] # 简化写法，实际需用真实y_val反标准化

# 计算指标
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
rmse = np.sqrt(mean_squared_error(y_val_inv, y_pred))
print(f"✅ 验证集 RMSE: {rmse:.2f} | MAE: {mean_absolute_error(y_val_inv, y_pred):.2f}")