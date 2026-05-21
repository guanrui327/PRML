import torch

class Config:
    # 数据设置
    SEQ_LEN = 10
    VOCAB_SIZE = 8          # 降低词汇量，随机基线=12.5%，现象更明显
    N_SAMPLES_TRAIN = 2000
    N_SAMPLES_VAL = 500
    BATCH_SIZE = 64

    # 模型设置
    D_MODEL = 128
    N_LAYERS = 2
    N_HEADS = 4
    D_FF = 256
    DROPOUT = 0.0           # 🔥 消融实验必须关闭，避免掩盖信号
    MAX_LEN = 50

    # 训练设置
    EPOCHS = 30
    LR = 5e-3               # 提高学习率加速收敛
    PATIENCE = 6            # 放宽早停
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    PE_TYPES = ['sinusoidal', 'learnable', 'none']
    TASK = 'pos_pattern'    # ✅ 切换为新任务