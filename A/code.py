import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# ------------------------------
# 1. 数据加载
# ------------------------------
def load_data(file_path):
    """读取 Excel 文件，返回训练和测试数据 (x, y)"""
    xls = pd.ExcelFile(file_path)
    # 假设第一个 sheet 是训练集，第二个 sheet 是测试集
    train_df = pd.read_excel(xls, sheet_name=0)
    test_df = pd.read_excel(xls, sheet_name=1)
    
    # 假设数据有两列，第一列为 x，第二列为 y（可根据实际列名调整）
    # 如果列名未知，使用 iloc 取前两列
    train_x = train_df.iloc[:, 0].values.astype(float)
    train_y = train_df.iloc[:, 1].values.astype(float)
    test_x = test_df.iloc[:, 0].values.astype(float)
    test_y = test_df.iloc[:, 1].values.astype(float)
    
    return train_x, train_y, test_x, test_y

# ------------------------------
# 2. 模型评估
# ------------------------------
def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def predict(w, b, x):
    return w * x + b

# ------------------------------
# 3. 最小二乘法（解析解）
# ------------------------------
def least_squares_fit(x, y):
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sum((x - x_mean) ** 2)
    w = numerator / denominator
    b = y_mean - w * x_mean
    return w, b

# ------------------------------
# 4. 梯度下降法
# ------------------------------
def gradient_descent_fit(x, y, lr=0.01, epochs=1000, verbose=False):
    m = len(x)
    w, b = 0.0, 0.0
    for epoch in range(epochs):
        y_pred = w * x + b
        dw = (1/m) * np.sum((y_pred - y) * x)
        db = (1/m) * np.sum(y_pred - y)
        w -= lr * dw
        b -= lr * db
        if verbose and epoch % 200 == 0:
            loss = mse(y, y_pred)
            print(f"GD epoch {epoch}, loss={loss:.6f}")
    return w, b

# ------------------------------
# 5. 牛顿法（解析二阶导数）
# ------------------------------
def newton_method_fit(x, y, max_iter=20, tol=1e-6, verbose=False):
    """
    对于线性回归损失函数 J = 1/(2m) * Σ(w*x_i + b - y_i)^2，
    梯度向量 g = [∂J/∂w, ∂J/∂b]^T，
    海森矩阵 H = [[∂²J/∂w², ∂²J/∂w∂b], [∂²J/∂b∂w, ∂²J/∂b²]]。
    由于损失函数为二次型，牛顿法一步即可收敛（理论上），这里仍写为迭代形式。
    """
    m = len(x)
    w, b = 0.0, 0.0  # 初始值
    for i in range(max_iter):
        y_pred = w * x + b
        # 梯度
        dw = (1/m) * np.sum((y_pred - y) * x)
        db = (1/m) * np.sum(y_pred - y)
        # 海森矩阵元素
        H_ww = (1/m) * np.sum(x * x)
        H_wb = (1/m) * np.sum(x)
        H_bb = 1.0  # (1/m) * Σ1 = 1
        H = np.array([[H_ww, H_wb], [H_wb, H_bb]])
        grad = np.array([dw, db])
        # 求解更新量 delta = - H^{-1} * grad
        try:
            delta = np.linalg.solve(H, -grad)
        except np.linalg.LinAlgError:
            # 若海森矩阵奇异，使用伪逆
            delta = -np.linalg.pinv(H) @ grad
        w += delta[0]
        b += delta[1]
        if verbose:
            print(f"Newton iter {i+1}, grad_norm={np.linalg.norm(grad):.6f}")
        if np.linalg.norm(grad) < tol:
            break
    return w, b

# ------------------------------
# 6. 可视化对比
# ------------------------------
def plot_models(train_x, train_y, test_x, test_y, models):
    """
    models: 字典，键为模型名称，值为 (w, b)
    """
    plt.figure(figsize=(10, 6))
    # 训练数据散点图
    plt.scatter(train_x, train_y, color='blue', alpha=0.6, label='Training data')
    # 测试数据散点图（可选，用不同标记）
    plt.scatter(test_x, test_y, color='green', alpha=0.4, marker='x', label='Test data')
    
    # 绘制回归线
    x_line = np.linspace(min(train_x.min(), test_x.min()), 
                         max(train_x.max(), test_x.max()), 100)
    colors = ['red', 'orange', 'purple']
    for (name, (w, b)), color in zip(models.items(), colors):
        y_line = w * x_line + b
        plt.plot(x_line, y_line, color=color, linewidth=2, label=f"{name}: y={w:.3f}x+{b:.3f}")
    
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Linear Regression: Least Squares vs GD vs Newton')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

# ------------------------------
# 7. 主程序
# ------------------------------
if __name__ == "__main__":
    # 请确保 Data4Regression.xlsx 在当前目录下
    file_path = "Data4Regression.xlsx"
    try:
        train_x, train_y, test_x, test_y = load_data(file_path)
    except Exception as e:
        print("读取文件失败，请检查文件路径和格式。")
        print(e)
        exit()
    
    print("数据加载成功")
    print(f"训练集样本数: {len(train_x)}")
    print(f"测试集样本数: {len(test_x)}")
    
    # 最小二乘法
    w_ls, b_ls = least_squares_fit(train_x, train_y)
    train_pred_ls = predict(w_ls, b_ls, train_x)
    test_pred_ls = predict(w_ls, b_ls, test_x)
    train_mse_ls = mse(train_y, train_pred_ls)
    test_mse_ls = mse(test_y, test_pred_ls)
    
    # 梯度下降法
    w_gd, b_gd = gradient_descent_fit(train_x, train_y, lr=0.01, epochs=1000, verbose=False)
    train_pred_gd = predict(w_gd, b_gd, train_x)
    test_pred_gd = predict(w_gd, b_gd, test_x)
    train_mse_gd = mse(train_y, train_pred_gd)
    test_mse_gd = mse(test_y, test_pred_gd)
    
    # 牛顿法
    w_nt, b_nt = newton_method_fit(train_x, train_y, max_iter=20, verbose=False)
    train_pred_nt = predict(w_nt, b_nt, train_x)
    test_pred_nt = predict(w_nt, b_nt, test_x)
    train_mse_nt = mse(train_y, train_pred_nt)
    test_mse_nt = mse(test_y, test_pred_nt)
    
    # 输出结果
    print("\n================ 模型参数 ================")
    print(f"最小二乘法: w = {w_ls:.6f}, b = {b_ls:.6f}")
    print(f"梯度下降法: w = {w_gd:.6f}, b = {b_gd:.6f}")
    print(f"牛顿法:     w = {w_nt:.6f}, b = {b_nt:.6f}")
    
    print("\n================ 训练误差 (MSE) ================")
    print(f"最小二乘法: {train_mse_ls:.8f}")
    print(f"梯度下降法: {train_mse_gd:.8f}")
    print(f"牛顿法:     {train_mse_nt:.8f}")
    
    print("\n================ 测试误差 (MSE) ================")
    print(f"最小二乘法: {test_mse_ls:.8f}")
    print(f"梯度下降法: {test_mse_gd:.8f}")
    print(f"牛顿法:     {test_mse_nt:.8f}")
    
    # 可视化
    models_dict = {
        "Least Squares": (w_ls, b_ls),
        "Gradient Descent": (w_gd, b_gd),
        "Newton Method": (w_nt, b_nt)
    }
    plot_models(train_x, train_y, test_x, test_y, models_dict)