import matplotlib.pyplot as plt
import torch
import os

def plot_curves(results_dict, metric='val_acc', save_dir='results'):
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(10, 6))
    for pe_type, (losses, accs, best) in results_dict.items():
        data = accs if metric == 'val_acc' else losses
        plt.plot(data, marker='o' if metric=='val_acc' else 's', 
                 label=f'{pe_type} (Best: {best:.3f})')
    plt.xlabel('Epoch')
    plt.ylabel(metric.upper())
    plt.title(f'Transformer Ablation: {metric.replace("_", " ").title()}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{metric}.png'), dpi=300)
    print(f"📊 曲线已保存至 {save_dir}/{metric}.png")

def plot_attention(attn_tensor, pe_type, task, save_dir='results'):
    os.makedirs(save_dir, exist_ok=True)
    attn = attn_tensor.numpy()
    plt.figure(figsize=(6, 5))
    im = plt.imshow(attn, cmap='viridis', aspect='auto')
    plt.colorbar(im, label='Attention Weight')
    plt.xlabel('Key Position')
    plt.ylabel('Query Position')
    plt.title(f'Attention Map: PE={pe_type} | Task={task}')
    plt.tight_layout()
    save_path = os.path.join(save_dir, f'attn_{pe_type}.png')
    plt.savefig(save_path, dpi=300)
    print(f"🎨 注意力图已保存至 {save_path}")