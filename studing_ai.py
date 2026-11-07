import torch
import torch.nn
import numpy as np
from PIL import Image
from ai_cnn import DoeNet

# 2.1. Константы
LAM, DX, N = 532e-9, 10e-6, 128
Z_LIST = [24.1e-3, 25.3e-3, 26.5e-3, 27.7e-3]

# 2.2. Данные
targets_np = [np.asarray(Image.open(f"targets/target_{i}.png"), np.float32)/255
              for i in range(4)]
targets_t = [torch.tensor(t) for t in targets_np]
x_in = torch.stack(targets_t).unsqueeze(0)       # (1,4,128,128)

# 2.3. Физика (дифференцируемая)
def propagate_torch(phi, z): ...                 # ← из пункта 5

# 2.4. Loss
def total_loss(phi, targets):
    loss = 0.0
    for target, z in zip(targets, Z_LIST):
        K = propagate_torch(phi, z)
        I = K.abs()**2
        I = I / (I.max() + 1e-8)
        loss = loss + (1 - pearson(I, target)) + (1 - ssim_simple(I, target))
    return loss / len(Z_LIST)

# 2.5. Устройство
device = "cuda" if torch.cuda.is_available() else "cpu"
x_in = x_in.to(device)
targets_t = [t.to(device) for t in targets_t]
