"""
Обучение нейросети синтеза фазового ДОЭ (128×128) и сохранение результатов.

Использование:
    python train.py                     # 500 эпох, автоустройство
    python train.py --epochs 800 --lr 1e-3
"""

import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from scipy.stats import pearsonr
from skimage.metrics import structural_similarity as ssim_np

# ────────────────────────────────────────────────────────────
# Константы (СИ!)
# ────────────────────────────────────────────────────────────
LAM = 532e-9                                # длина волны, м
DX  = 10e-6                                 # пиксел ПВМС, м
N   = 128                                   # размер сетки
Z_LIST = [24.1e-3, 25.3e-3, 26.5e-3, 27.7e-3]   # расстояния, м
TARGET_DIR = "targets"


# ────────────────────────────────────────────────────────────
# Физика: propagate (numpy) и propagate_torch (дифференцируемо)
# ────────────────────────────────────────────────────────────
def propagate(phi, z, lam=LAM, dx=DX):
    """numpy-версия — для финальных картинок."""
    N_ = phi.shape[0]
    j = 1j
    k = 2 * np.pi / lam
    u = (np.arange(N_) - N_ // 2) * dx
    U, V = np.meshgrid(u, u, indexing="xy")

    H = np.exp(j * phi)
    chirp_in = np.exp(j * np.pi / (lam * z) * (U**2 + V**2))
    F = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(H * chirp_in)))

    dx_out = lam * z / (N_ * dx)
    x = (np.arange(N_) - N_ // 2) * dx_out
    X, Y = np.meshgrid(x, x, indexing="xy")

    A = np.exp(j * k * z) / (j * lam * z)
    chirp_out = np.exp(j * np.pi / (lam * z) * (X**2 + Y**2))
    return A * chirp_out * F


def propagate_torch(phi, z, lam=LAM, dx=DX):
    """torch-версия — для обучения (градиенты текут)."""
    squeeze = (phi.dim() == 2)
    if squeeze:
        phi = phi.unsqueeze(0)

    _, N_, _ = phi.shape
    dev = phi.device
    j = torch.tensor(1j, dtype=torch.complex64, device=dev)
    k = 2 * torch.pi / lam

    u = (torch.arange(N_, device=dev) - N_ // 2).float() * dx
    U, V = torch.meshgrid(u, u, indexing="xy")

    H = torch.exp(j * phi.squeeze(1).to(torch.complex64))
    chirp_in = torch.exp(j * torch.pi / (lam * z) * (U**2 + V**2))
    F = torch.fft.fftshift(
        torch.fft.fft2(torch.fft.ifftshift(H * chirp_in, dim=(-2, -1)), dim=(-2, -1)),
        dim=(-2, -1),
    )

    dx_out = lam * z / (N_ * dx)
    x = (torch.arange(N_, device=dev) - N_ // 2).float() * dx_out
    X, Y = torch.meshgrid(x, x, indexing="xy")

    A = torch.exp(j * k * z) / (j * lam * z)
    chirp_out = torch.exp(j * torch.pi / (lam * z) * (X**2 + Y**2))

    K = A * chirp_out * F
    return K.squeeze(0) if squeeze else K


# ────────────────────────────────────────────────────────────
# Метрики
# ────────────────────────────────────────────────────────────
def pearson(a, b, eps=1e-8):
    a = a - a.mean(dim=(-2, -1), keepdim=True)
    b = b - b.mean(dim=(-2, -1), keepdim=True)
    num = (a * b).sum(dim=(-2, -1))
    den = a.norm(dim=(-2, -1)) * b.norm(dim=(-2, -1)) + eps
    return (num / den).mean()


def ssim_simple(a, b, eps=1e-8):
    mu_a = a.mean(dim=(-2, -1), keepdim=True)
    mu_b = b.mean(dim=(-2, -1), keepdim=True)
    var_a = a.var(dim=(-2, -1), unbiased=False, keepdim=True)
    var_b = b.var(dim=(-2, -1), unbiased=False, keepdim=True)
    cov = ((a - mu_a) * (b - mu_b)).mean(dim=(-2, -1), keepdim=True)
    C1, C2 = 0.01**2, 0.03**2
    ssim_map = ((2*mu_a*mu_b + C1) * (2*cov + C2)) / \
               ((mu_a**2 + mu_b**2 + C1) * (var_a + var_b + C2))
    return ssim_map.mean()


def total_loss(phi, targets):
    if phi.dim() == 3:
        phi = phi.squeeze(0)
    loss = 0.0
    for target, z in zip(targets, Z_LIST):
        K = propagate_torch(phi, z)
        I = K.abs() ** 2
        I = I / (I.max() + 1e-8)
        loss = loss + (1 - pearson(I, target)) + (1 - ssim_simple(I, target))
    return loss / len(Z_LIST)


# ────────────────────────────────────────────────────────────
# Сеть
# ────────────────────────────────────────────────────────────
class DoeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4, 32, 3, padding=1),  nn.BatchNorm2d(32),  nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64),  nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=1), nn.BatchNorm2d(64),  nn.ReLU(),
            nn.Conv2d(64, 32, 3, padding=1),  nn.BatchNorm2d(32),  nn.ReLU(),
            nn.Conv2d(32, 1, 3, padding=1),
        )

    def forward(self, x):
        return torch.tanh(self.net(x)) * torch.pi


# ────────────────────────────────────────────────────────────
# Загрузка данных
# ────────────────────────────────────────────────────────────
def load_targets(dir_=TARGET_DIR):
    paths = [os.path.join(dir_, f"target_{i}.png") for i in range(4)]
    for p in paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Нет файла {p}. Сначала запустите prepare.py")
    arrs = [np.asarray(Image.open(p), dtype=np.float32) / 255.0 for p in paths]
    return arrs


# ────────────────────────────────────────────────────────────
# Обучение
# ────────────────────────────────────────────────────────────
def train(epochs=500, lr=1e-3, log_every=25):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Устройство: {device}")

    targets_np = load_targets()
    targets_t = [torch.tensor(t, device=device) for t in targets_np]
    x_in = torch.stack(targets_t).unsqueeze(0)          # (1,4,128,128)

    model = DoeNet().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Параметров: {n_params:,}")

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    t0 = time.time()
    best_loss = float("inf")
    for epoch in range(1, epochs + 1):
        model.train()
        phi = model(x_in).squeeze()                     # (128,128)
        loss = total_loss(phi, targets_t)

        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            torch.save(model.state_dict(), "model_best.pth")

        if epoch % log_every == 0 or epoch == 1:
            # метрики для лога
            with torch.no_grad():
                model.eval()
                phi_eval = model(x_in).squeeze()
                kk = np.mean([
                    pearson(propagate_torch(phi_eval, z).abs() ** 2 /
                            (propagate_torch(phi_eval, z).abs() ** 2).max() + 1e-8,
                            t).item()
                    for t, z in zip(targets_t, Z_LIST)
                ])
            dt = time.time() - t0
            print(f"epoch {epoch:4d} | loss {loss.item():.4f} | "
                  f"КК(ср) {kk:.4f} | {dt:.1f}s")

    torch.save(model.state_dict(), "model.pth")
    print(f"\nОбучение завершено за {time.time()-t0:.1f}s")
    print(f"model.pth сохранён (best loss = {best_loss:.4f})")
    return model, targets_np


# ────────────────────────────────────────────────────────────
# Сохранение результатов
# ────────────────────────────────────────────────────────────
def save_results(model, targets_np):
    device = next(model.parameters()).device
    model.eval()

    targets_t = [torch.tensor(t, device=device) for t in targets_np]
    x_in = torch.stack(targets_t).unsqueeze(0)

    with torch.no_grad():
        phi = model(x_in).squeeze().cpu().numpy()       # (128,128)

    # 1. ДОЭ в PNG: 0 ↔ -π, 255 ↔ +π
    doe = ((phi + np.pi) / (2 * np.pi) * 255).round().astype(np.uint8)
    Image.fromarray(doe).save("doe.png")
    print("Сохранено: doe.png")

    # 2. 4 восстановления
    rows = []
    for i, z in enumerate(Z_LIST):
        K = propagate(phi, z)
        I = np.abs(K) ** 2
        I = I / (I.max() + 1e-8)
        Image.fromarray((I * 255).astype(np.uint8)).save(f"recovered_{i}.png")

        # метрики
        t = targets_np[i]
        r = I.astype(np.float32)
        kk = pearsonr(t.ravel(), r.ravel())[0]
        iss = ssim_np(t, r, data_range=1.0)
        rows.append((i, z * 1e3, kk, iss))
        print(f"recovered_{i}.png  (z={z*1e3:.1f} мм)  "
              f"КК={kk:.4f}  ИСС={iss:.4f}")

    # 3. metrics.txt
    with open("metrics.txt", "w") as f:
        f.write("img\tz_mm\tKK\tISS\n")
        for i, z_mm, kk, iss in rows:
            f.write(f"{i}\t{z_mm:.1f}\t{kk:.4f}\t{iss:.4f}\n")
    print("Сохранено: metrics.txt")

    # 4. средние
    kk_mean = np.mean([r[2] for r in rows])
    iss_mean = np.mean([r[3] for r in rows])
    print(f"\nСредние: КК={kk_mean:.4f}  ИСС={iss_mean:.4f}")


# ────────────────────────────────────────────────────────────
# main
# ────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--skip-train", action="store_true",
                    help="только сгенерировать результаты из model.pth")
    args = ap.parse_args()

    if args.skip_train:
        if not os.path.exists("model.pth"):
            print("Нет model.pth — сначала обучите.")
            return
        model = DoeNet()
        model.load_state_dict(torch.load("model.pth", map_location="cpu"))
        targets_np = load_targets()
        save_results(model, targets_np)
        return

    model, targets_np = train(args.epochs, args.lr, args.log_every)
    save_results(model, targets_np)


if __name__ == "__main__":
    main()
