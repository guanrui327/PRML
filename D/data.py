import torch
from torch.utils.data import DataLoader, TensorDataset
from config import Config

def generate_task(task_type='pos_pattern', seq_len=None, vocab_size=None, n_samples=None):
    seq_len = seq_len or Config.SEQ_LEN
    vocab_size = vocab_size or Config.VOCAB_SIZE
    n_samples = n_samples or Config.N_SAMPLES_TRAIN

    if task_type == 'pos_pattern':
        # 输入全0，彻底隔离内容信息，强制模型依赖位置编码
        X = torch.zeros(n_samples, seq_len, dtype=torch.long)
        # 目标严格由位置索引决定：0,1,2,3...
        Y = (torch.arange(seq_len).unsqueeze(0) % vocab_size).repeat(n_samples, 1)
    elif task_type == 'copy':
        X = torch.randint(0, vocab_size, (n_samples, seq_len))
        Y = X.clone()
    elif task_type == 'reverse':
        X = torch.randint(0, vocab_size, (n_samples, seq_len))
        Y = torch.flip(X, dims=[1])
    else:
        raise ValueError("task_type must be 'pos_pattern', 'copy', or 'reverse'")
    return X, Y

def get_dataloaders(task_type='pos_pattern', batch_size=None):
    batch_size = batch_size or Config.BATCH_SIZE
    X_train, y_train = generate_task(task_type, n_samples=Config.N_SAMPLES_TRAIN)
    X_val, y_val = generate_task(task_type, n_samples=Config.N_SAMPLES_VAL)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False)
    return train_loader, val_loader