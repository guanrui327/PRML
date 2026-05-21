# train.py (仅替换 run_ablation 中的调度器部分)
import torch
import torch.nn as nn
from tqdm import tqdm
from config import Config
from data import get_dataloaders
from model import TransformerEncoder

def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for src, tgt in tqdm(dataloader, desc="Training", leave=False):
        src, tgt = src.to(device), tgt.to(device)
        optimizer.zero_grad()
        output = model(src)
        loss = criterion(output.view(-1, output.size(-1)), tgt.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate(model, dataloader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for src, tgt in tqdm(dataloader, desc="Evaluating", leave=False):
            src, tgt = src.to(device), tgt.to(device)
            output = model(src)
            pred = output.argmax(dim=-1)
            correct += (pred == tgt).sum().item()
            total += tgt.numel()
    return correct / total

def run_ablation(pe_type, task_type='reverse'):
    device = Config.DEVICE
    train_loader, val_loader = get_dataloaders(task_type)

    model = TransformerEncoder(
        vocab_size=Config.VOCAB_SIZE, d_model=Config.D_MODEL,
        n_layers=Config.N_LAYERS, num_heads=Config.N_HEADS,
        d_ff=Config.D_FF, max_len=Config.MAX_LEN,
        pe_type=pe_type, dropout=Config.DROPOUT
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LR, betas=(0.9, 0.98), eps=1e-9)
    # ✅ 改用余弦衰减，避免过早降学习率
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.EPOCHS)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    patience_counter = 0
    train_losses, val_accs = [], []

    print(f"\n🚀 实验启动: PE={pe_type} | Task={task_type}")
    for epoch in range(Config.EPOCHS):
        tr_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_acc = evaluate(model, val_loader, device)
        scheduler.step()

        train_losses.append(tr_loss)
        val_accs.append(val_acc)

        print(f"Epoch {epoch+1:02d} | Loss: {tr_loss:.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), f'best_{pe_type}_{task_type}.pth')
            model.eval()
            with torch.no_grad():
                sample_src, _ = next(iter(val_loader))
                sample_src = sample_src.to(device)
                _ = model(sample_src)
                torch.save(model.layers[0].self_attn.attn_weights[0, 0].cpu(), f'attn_{pe_type}_{task_type}.pt')
        else:
            patience_counter += 1
            if patience_counter >= Config.PATIENCE:
                print(f"⏹️ 早停触发于 Epoch {epoch+1}")
                break

    return train_losses, val_accs, best_acc