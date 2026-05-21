"""
Transformer位置编码消融实验
任务：序列复制与反转（验证位置编码必要性）
作者：管睿
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import random

# 设置随机种子
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

set_seed(42)

# ==================== 1. 位置编码模块 ====================

class SinusoidalPositionalEncoding(nn.Module):
    """正弦/余弦位置编码（Transformer原文）"""
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        # x: (batch, seq_len, d_model)
        return x + self.pe[:, :x.size(1), :]

class LearnablePositionalEncoding(nn.Module):
    """可学习绝对位置编码"""
    def __init__(self, d_model, max_len=100):
        super().__init__()
        self.pe = nn.Parameter(torch.randn(1, max_len, d_model) * 0.1)
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class NoPositionalEncoding(nn.Module):
    """无位置编码（消融对照）"""
    def __init__(self, d_model, max_len=100):
        super().__init__()
    
    def forward(self, x):
        return x

# ==================== 2. Transformer模型 ====================

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.size()
        
        Q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        out = self.W_o(out)
        return out

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
    
    def forward(self, x):
        return self.linear2(self.dropout(self.activation(self.linear1(x))))

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # Self-attention with residual
        attn_out = self.attention(x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        # FFN with residual
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, d_ff, max_len, 
                 dropout=0.1, pos_encoding='sinusoidal'):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.dropout = nn.Dropout(dropout)
        
        # 选择位置编码
        if pos_encoding == 'sinusoidal':
            self.pos_encoder = SinusoidalPositionalEncoding(d_model, max_len)
        elif pos_encoding == 'learnable':
            self.pos_encoder = LearnablePositionalEncoding(d_model, max_len)
        elif pos_encoding == 'none':
            self.pos_encoder = NoPositionalEncoding(d_model, max_len)
        else:
            raise ValueError(f"Unknown pos_encoding: {pos_encoding}")
        
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout) 
            for _ in range(n_layers)
        ])
        
        self.fc_out = nn.Linear(d_model, vocab_size)
        
    def forward(self, x, mask=None):
        # Embedding
        x = self.embedding(x)
        x = self.dropout(x)
        # Positional Encoding
        x = self.pos_encoder(x)
        # Transformer blocks
        for layer in self.layers:
            x = layer(x, mask)
        # Output projection
        logits = self.fc_out(x)
        return logits

# ==================== 3. 序列复制/反转任务数据生成 ====================

def generate_sequence_data(num_samples, seq_len, vocab_size, task='copy'):
    """
    生成序列任务数据
    task: 'copy' - 复制任务，'reverse' - 反转任务
    """
    # 确保vocab中有特殊token: 0=PAD, 1=SOS, 2=EOS
    # 实际数据从3开始
    data = np.random.randint(3, vocab_size, size=(num_samples, seq_len))
    
    if task == 'copy':
        targets = data.copy()
    elif task == 'reverse':
        targets = np.flip(data, axis=1).copy()
    else:
        raise ValueError(f"Unknown task: {task}")
    
    return torch.LongTensor(data), torch.LongTensor(targets)

# ==================== 4. 训练和评估 ====================

def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        output = model(x)
        # 计算每个位置的交叉熵损失
        loss = criterion(output.reshape(-1, output.size(-1)), y.reshape(-1))
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            loss = criterion(output.reshape(-1, output.size(-1)), y.reshape(-1))
            total_loss += loss.item()
            
            pred = output.argmax(dim=-1)
            correct += (pred == y).sum().item()
            total += y.numel()
    
    return total_loss / len(dataloader), correct / total

def run_experiment(pos_encoding, task, seq_len=10, d_model=64, n_heads=4, 
                   n_layers=2, d_ff=128, batch_size=64, epochs=100, 
                   lr=0.001, device='cuda'):
    """
    运行单次实验
    """
    print(f"\n{'='*60}")
    print(f"实验配置: 位置编码={pos_encoding}, 任务={task}")
    print(f"{'='*60}")
    
    # 参数设置
    vocab_size = 20  # 0-19, 其中0=PAD, 1=SOS, 2=EOS, 3-19为数据
    max_len = seq_len + 2
    
    # 生成数据
    train_size = 8000
    val_size = 1000
    test_size = 1000
    
    train_x, train_y = generate_sequence_data(train_size, seq_len, vocab_size, task)
    val_x, val_y = generate_sequence_data(val_size, seq_len, vocab_size, task)
    test_x, test_y = generate_sequence_data(test_size, seq_len, vocab_size, task)
    
    train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_x, val_y), batch_size=batch_size)
    test_loader = DataLoader(TensorDataset(test_x, test_y), batch_size=batch_size)
    
    # 创建模型
    model = TransformerEncoder(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        max_len=max_len,
        dropout=0.1,
        pos_encoding=pos_encoding
    ).to(device)
    
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # 忽略PAD
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # 训练记录
    train_losses = []
    val_losses = []
    val_accs = []
    best_val_loss = float('inf')
    
    for epoch in tqdm(range(epochs), desc=f"Training {pos_encoding}"):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
        
        if (epoch + 1) % 20 == 0:
            print(f"\nEpoch {epoch+1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")
    
    # 测试最佳模型
    model.load_state_dict(best_model_state)
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    
    print(f"\n最终测试结果: loss={test_loss:.4f}, accuracy={test_acc:.4f}")
    
    return {
        'pos_encoding': pos_encoding,
        'task': task,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_accs': val_accs,
        'test_loss': test_loss,
        'test_acc': test_acc,
        'best_val_loss': best_val_loss
    }

# ==================== 5. 主实验 ====================

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 实验配置
    pos_encodings = ['sinusoidal', 'learnable', 'none']
    tasks = ['copy', 'reverse']
    
    results = {}
    
    for task in tasks:
        for pos_enc in pos_encodings:
            key = f"{task}_{pos_enc}"
            results[key] = run_experiment(
                pos_encoding=pos_enc,
                task=task,
                seq_len=8,
                d_model=64,
                n_heads=4,
                n_layers=2,
                d_ff=128,
                batch_size=64,
                epochs=80,
                device=device
            )
    
    # ==================== 可视化结果 ====================
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    for idx, task in enumerate(tasks):
        for jdx, pos_enc in enumerate(pos_encodings):
            key = f"{task}_{pos_enc}"
            res = results[key]
            
            ax = axes[idx, jdx]
            epochs = range(1, len(res['val_accs']) + 1)
            ax.plot(epochs, res['val_accs'], label='Validation Accuracy', linewidth=2)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Accuracy')
            ax.set_title(f'{task.capitalize()} - {pos_enc.capitalize()} PE\nTest Acc: {res["test_acc"]:.4f}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('positional_encoding_comparison.png', dpi=150)
    plt.show()
    
    # 打印总结表格
    print("\n" + "="*80)
    print("实验结果总结")
    print("="*80)
    print(f"{'任务':<12} {'位置编码':<15} {'测试准确率':<12} {'最佳验证损失':<12}")
    print("-"*80)
    
    for task in tasks:
        for pos_enc in pos_encodings:
            key = f"{task}_{pos_enc}"
            res = results[key]
            print(f"{task:<12} {pos_enc:<15} {res['test_acc']:.4f}       {res['best_val_loss']:.6f}")
    
    print("="*80)
    
    # 保存结果
    torch.save(results, 'experiment_results.pt')
    print("\n实验结果已保存至 experiment_results.pt")
    
    return results

if __name__ == "__main__":
    results = main()