# main.py
import os
import torch  # ✅ 修复 NameError
from config import Config
from train import run_ablation
from utils import plot_curves, plot_attention

def main():
    os.makedirs('results', exist_ok=True)
    results = {}
    task = Config.TASK

    print(f"🔍 开始消融实验 | 任务: {task} | 设备: {Config.DEVICE}")
    for pe_type in Config.PE_TYPES:
        losses, accs, best_acc = run_ablation(pe_type, task)
        results[pe_type] = (losses, accs, best_acc)

    print("\n📊 实验结果汇总:")
    print(f"{'位置编码类型':<15} | {'最佳验证集Token准确率':<15}")
    print("-" * 40)
    for pe, (_, _, best) in results.items():
        print(f"{pe:<15} | {best:.4f}")

    # 绘制对比曲线
    plot_curves(results, metric='val_acc')
    plot_curves(results, metric='loss')

    # 绘制注意力权重可视化
    for pe_type in Config.PE_TYPES:
        attn_file = f'attn_{pe_type}_{task}.pt'
        if os.path.exists(attn_file):
            attn = torch.load(attn_file, map_location='cpu')  # ✅ 避免新版本警告
            plot_attention(attn, pe_type, task)

    print("\n✅ 实验全部完成！图表与权重已保存至 results/ 目录。")

if __name__ == "__main__":
    main()