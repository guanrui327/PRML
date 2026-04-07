import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import warnings
warnings.filterwarnings('ignore')

def make_moons_3d(n_samples=500, noise=0.1):
    t = np.linspace(0, 2 * np.pi, n_samples)
    x = 1.5 * np.cos(t)
    y = np.sin(t)
    z = np.sin(2 * t)
    X = np.vstack([np.column_stack([x, y, z]), np.column_stack([-x, y - 1, -z])])
    y_labels = np.hstack([np.zeros(n_samples), np.ones(n_samples)])
    X += np.random.normal(scale=noise, size=X.shape)
    return X, y_labels

np.random.seed(42)
X_train, y_train = make_moons_3d(n_samples=500, noise=0.2)

np.random.seed(2026)
X_test, y_test = make_moons_3d(n_samples=250, noise=0.2)


scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

models = {
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
    "AdaBoost + DT": AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=3, random_state=42),
        n_estimators=100, learning_rate=1.0, random_state=42
    ),
    "SVM (Linear)": SVC(kernel='linear', C=1.0, random_state=42),
    "SVM (Poly)":   SVC(kernel='poly', degree=3, C=1.0, gamma='scale', random_state=42),
    "SVM (RBF)":    SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
}

models["AdaBoost + DT"] = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=3, random_state=42),
    n_estimators=100, learning_rate=1.0, random_state=42
)

trained_models = {}
for name, clf in models.items():
    clf.fit(X_train_sc, y_train)
    trained_models[name] = clf

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(X_test[:, 0], X_test[:, 1], X_test[:, 2], 
                     c=y_test, cmap='viridis', s=20, alpha=0.6, edgecolors='k')
ax.set_xlabel('X', fontsize=10)
ax.set_ylabel('Y', fontsize=10)
ax.set_zlabel('Z', fontsize=10)
ax.set_title('📊 测试集原始分布 (3D)', fontsize=12, fontweight='bold')
legend = ax.legend(*scatter.legend_elements(), title="Classes", loc='upper right')
ax.add_artist(legend)
plt.tight_layout()
plt.savefig('3d_data_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

def plot_2d_decision_boundary(model, X_test, y_test, feature_pair=(0,1), 
                              feature_names=['X','Y','Z'], model_name="Model", n_points=200):
    """绘制2D投影平面的决策边界等高线"""
    f1, f2 = feature_pair
    x_min, x_max = X_test[:, f1].min() - 1, X_test[:, f1].max() + 1
    y_min, y_max = X_test[:, f2].min() - 1, X_test[:, f2].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, n_points),
                         np.linspace(y_min, y_max, n_points))
    
    
    fixed_dim = np.mean(X_test[:, [i for i in range(3) if i not in [f1, f2]]])
    if f1 == 0 and f2 == 1:
        grid = np.column_stack([xx.ravel(), yy.ravel(), np.full(len(xx.ravel()), fixed_dim)])
    elif f1 == 0 and f2 == 2:
        grid = np.column_stack([xx.ravel(), np.full(len(xx.ravel()), fixed_dim), yy.ravel()])
    else:  # 1,2
        grid = np.column_stack([np.full(len(xx.ravel()), fixed_dim), xx.ravel(), yy.ravel()])
    
    grid_sc = scaler.transform(grid)  # 重要：网格点也要标准化！
    
    # 预测决策函数值（用于等高线）
    if hasattr(model, 'decision_function'):
        Z = model.decision_function(grid_sc)
    else:
        Z = model.predict_proba(grid_sc)[:, 1]
    Z = Z.reshape(xx.shape)
    
    
    plt.figure(figsize=(6, 5))
    contour = plt.contourf(xx, yy, Z, levels=20, cmap='RdYlBu_r', alpha=0.3)
    plt.scatter(X_test[:, f1], X_test[:, f2], c=y_test, cmap='viridis', 
                s=20, edgecolors='k', alpha=0.7)
    plt.xlabel(feature_names[f1], fontsize=10)
    plt.ylabel(feature_names[f2], fontsize=10)
    plt.title(f'🗺️ {model_name} 决策边界 ({feature_names[f1]}-{feature_names[f2]}投影)', 
              fontsize=11, fontweight='bold')
    plt.colorbar(contour, label='Decision Function / P(C1)')
    plt.tight_layout()
    plt.savefig(f'decision_boundary_{model_name.replace(" ", "_")}_{feature_names[f1]}{feature_names[f2]}.png', 
                dpi=300, bbox_inches='tight')
    plt.show()

best_model = trained_models["SVM (RBF)"]
for pair, names in zip([(0,1), (0,2), (1,2)], [('X','Y'), ('X','Z'), ('Y','Z')]):
    plot_2d_decision_boundary(best_model, X_test, y_test, feature_pair=pair, 
                              feature_names=['X','Y','Z'], model_name="SVM (RBF)")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.ravel()
model_names = list(trained_models.keys())

for idx, (name, clf) in enumerate(trained_models.items()):
    y_pred = clf.predict(X_test_sc)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['C0', 'C1'])
    disp.plot(ax=axes[idx], cmap='Blues', colorbar=False, values_format='d')
    acc = np.trace(cm) / np.sum(cm)
    axes[idx].set_title(f'{name}\nAccuracy={acc:.3f}', fontsize=11, fontweight='bold')
    axes[idx].set_xlabel('Predicted', fontsize=9)
    axes[idx].set_ylabel('True', fontsize=9)

axes[-1].axis('off')

plt.suptitle('🔍 各模型混淆矩阵对比 (测试集 N=500)', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('confusion_matrices_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

results = []
for name, clf in trained_models.items():
    y_pred = clf.predict(X_test_sc)
    cm = confusion_matrix(y_test, y_pred)
    acc = np.trace(cm) / np.sum(cm)
    # 计算 macro F1
    from sklearn.metrics import f1_score
    f1 = f1_score(y_test, y_pred, average='macro')
    results.append({'Model': name, 'Accuracy': acc, 'F1-Macro': f1})

import pandas as pd
df_res = pd.DataFrame(results).set_index('Model')

fig, ax = plt.subplots(1, 2, figsize=(14, 5))

# Accuracy 
colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(df_res)))
bars1 = ax[0].barh(df_res.index, df_res['Accuracy'], color=colors, edgecolor='black')
ax[0].set_xlabel('Accuracy', fontsize=11)
ax[0].set_title('📈 分类准确率对比', fontsize=12, fontweight='bold')
ax[0].set_xlim(0.5, 1.0)
for bar, val in zip(bars1, df_res['Accuracy']):
    ax[0].text(val + 0.01, bar.get_y() + bar.get_height()/2, f'{val:.3f}', 
               va='center', fontsize=9)

# F1-Macro 
bars2 = ax[1].barh(df_res.index, df_res['F1-Macro'], color=colors, edgecolor='black')
ax[1].set_xlabel('Macro F1-Score', fontsize=11)
ax[1].set_title('⚖️ 类别均衡性 (F1) 对比', fontsize=12, fontweight='bold')
ax[1].set_xlim(0.5, 1.0)
for bar, val in zip(bars2, df_res['F1-Macro']):
    ax[1].text(val + 0.01, bar.get_y() + bar.get_height()/2, f'{val:.3f}', 
               va='center', fontsize=9)

plt.suptitle('🏆 模型性能综合对比', fontsize=14, fontweight='bold', y=1.05)
plt.tight_layout()
plt.savefig('performance_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

def plot_confidence_distribution(model, X_test, y_test, model_name):
    """绘制预测概率/决策函数值的分布直方图"""
    if hasattr(model, 'predict_proba'):
        probas = model.predict_proba(X_test)[:, 1]
        xlabel = 'P(Class C1)'
    else:
        probas = model.decision_function(X_test)
        xlabel = 'Decision Function Value'
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    

    for idx, label in enumerate([0, 1]):
        mask = y_test == label
        axes[idx].hist(probas[mask], bins=30, color=f'C{int(label)}', 
                       edgecolor='black', alpha=0.7)
        axes[idx].axvline(x=0.5 if 'P(' in xlabel else 0, color='red', 
                          linestyle='--', label='Decision Threshold')
        axes[idx].set_xlabel(xlabel, fontsize=10)
        axes[idx].set_ylabel('Count', fontsize=10)
        axes[idx].set_title(f'True Label = C{label}', fontsize=11, fontweight='bold')
        axes[idx].legend()
        axes[idx].grid(axis='y', alpha=0.3)
    
    plt.suptitle(f'🎯 {model_name} 分类置信度分布', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'confidence_{model_name.replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
    plt.show()

plot_confidence_distribution(best_model, X_test_sc, y_test, "SVM (RBF)")