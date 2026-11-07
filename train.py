"""
Обучение нейросети синтеза фазового ДОЭ и сохранение результатов.

Условия задачи (AI фотоника.pdf):
    - 4 объектных изображения 128×128 (8 бит, grayscale, png);
    - фазовый ДОЭ 128×128, значения пикселов -π..π, пиксел ПВМС 10 мкм;
    - восстановление прямым расчётом дифракции Френеля (одно FFT);
    - дистанции: 24.1 / 25.3 / 26.5 / 27.7 мм, λ = 532 нм;
    - метрики: коэффициент корреляции Пирсона (КК) и SSIM (ИСС).

Использование:
    python train.py                              # 500 эпох, автоустройство
    python train.py --epochs 1000 --lr 2e-3 --sched cosine
    python train.py --skip-train                 # только результаты из model.pth

Оптимизации скорости:
    - chirp-множители/meshgrid кэшируются один раз;
    - H = exp(j*phi) считается один раз на все z;
    - torch.polar вместо ручной сборки комплексных чисел.
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
LAM = 532e-9
DX  = 10e-6          # шаг пикселя ПВМС (по условию — 10 мкм)
N   = 128            # размер изображения и ДОЭ (по условию — 128×128)
Z_LIST = [24.1e-3, 25.3e-3, 26.5e-3, 27.7e-3]
TARGET_DIR = "targets"


# ────────────────────────────────────────────────────────────
# Физика (numpy — для диагностики test.py)
# ────────────────────────────────────────────────────────────
def propagate(phi, z, lam=LAM, dx=DX, pad_factor=2):
    N_orig = phi.shape[0]
    N_pad = N_orig * pad_factor
    off = (N_pad - N_orig) // 2

    j = 1j
    k = 2 * np.pi / lam

    u = (np.arange(N_pad) - N_pad // 2) * dx
    U, V = np.meshgrid(u, u, indexing="xy")

    phi_pad = np.zeros((N_pad, N_pad), dtype=phi.dtype)
    phi_pad[off:off + N_orig, off:off + N_orig] = phi

    H = np.exp(j * phi_pad)
    chirp_in = np.exp(j * np.pi / (lam * z) * (U**2 + V**2))
    F = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(H * chirp_in)))

    dx_out = lam * z / (N_pad * dx)
    x = (np.arange(N_pad) - N_pad // 2) * dx_out
    X, Y = np.meshgrid(x, x, indexing="xy")

    A = np.exp(j * k * z) / (j * lam * z)
    chirp_out = np.exp(j * np.pi / (lam * z) * (X**2 + Y**2))
    K_pad = A * chirp_out * F

    return K_pad[off:off + N_orig, off:off + N_orig]


# ────────────────────────────────────────────────────────────
# Физика (torch, дифференцируемая, оптимизированная)
# ────────────────────────────────────────────────────────────
class Propagator:
    """
    Дифференцируемое распространение Френеля для набора фиксированных z.
    Все chirp-множители и сетки предвычисляются ОДИН раз (на нужном device).
    H = exp(j*phi) вычисляется один раз и переиспользуется для всех z.

    Параметры:
        N        — размер (разрешение) ДОЭ (по условию 128).
        dx       — шаг пикселя ДОЭ.
        target_N — размер вырезаемой области (размер мишени, 128).
    """

    def __init__(self, z_list, lam=LAM, dx=DX, N=N, target_N=128,
                 pad_factor=2):
        self.z_list = z_list
        self.lam = lam
        self.N = N
        self.dx = dx
        self.target_N = target_N
        self.pad_factor = pad_factor
        self.N_pad = N * pad_factor
        self.off = (self.N_pad - N) // 2
        self._crop_off = (self.N_pad - target_N) // 2
        self._kernels = None
        self._dev = None

    def _prepare(self, device):
        if self._dev == device and self._kernels is not None:
            return
        N_pad = self.N_pad
        lam, dx = self.lam, self.dx
        k = 2 * torch.pi / lam

        u = (torch.arange(N_pad, device=device, dtype=torch.float32) - N_pad // 2) * dx
        U, V = torch.meshgrid(u, u, indexing="xy")
        r2 = U**2 + V**2

        chirp_in_list, chirp_out_list = [], []
        for z in self.z_list:
            chirp_in = torch.polar(torch.ones_like(r2), torch.pi / (lam * z) * r2)

            dx_out = lam * z / (N_pad * dx)
            x = (torch.arange(N_pad, device=device, dtype=torch.float32) - N_pad // 2) * dx_out
            X, Y = torch.meshgrid(x, x, indexing="xy")
            A = torch.exp(torch.tensor(1j, dtype=torch.complex64, device=device) * k * z) \
                / (torch.tensor(1j, dtype=torch.complex64, device=device) * lam * z)
            chirp_out = A * torch.polar(
                torch.ones_like(X), torch.pi / (lam * z) * (X**2 + Y**2)
            )
            chirp_in_list.append(chirp_in)
            chirp_out_list.append(chirp_out)

        self._kernels = (torch.stack(chirp_in_list), torch.stack(chirp_out_list))
        self._dev = device

    def propagate(self, phi):
        """
        phi: (N, N) или (B, N, N) — фаза ДОЭ.
        Возвращает комплексное поле (B, Z, target_N, target_N).
        """
        device = phi.device
        self._prepare(device)

        squeeze = phi.dim() == 2
        if squeeze:
            phi = phi.unsqueeze(0)
        B = phi.shape[0]
        Z = len(self.z_list)
        N_pad, off, co = self.N_pad, self.off, self._crop_off
        chirp_in, chirp_out = self._kernels

        phi_pad = torch.zeros(B, N_pad, N_pad, dtype=phi.dtype, device=device)
        phi_pad[:, off:off + self.N, off:off + self.N] = phi

        H = torch.polar(torch.ones_like(phi_pad), phi_pad)

        K_list = []
        for i in range(Z):
            F = torch.fft.fftshift(
                torch.fft.fft2(torch.fft.ifftshift(H * chirp_in[i], dim=(-2, -1)),
                                dim=(-2, -1)),
                dim=(-2, -1),
            )
            K = (F * chirp_out[i])[:, co:co + self.target_N, co:co + self.target_N]
            K_list.append(K)

        K = torch.stack(K_list, dim=1)                     # (B, Z, target_N, target_N)
        return K.squeeze(0) if squeeze else K


# ────────────────────────────────────────────────────────────
# Совместимость: одиночное распространение (для test.py)
# ────────────────────────────────────────────────────────────
_single_prop_cache = {}
_default_prop = None
_default_prop_device = None


def _get_default_prop(device):
    global _default_prop, _default_prop_device
    if _default_prop is None or _default_prop_device != device:
        _default_prop = Propagator(Z_LIST)
        _default_prop._prepare(device)
        _default_prop_device = device
    return _default_prop


def propagate_torch(phi, z, lam=LAM, dx=DX, pad_factor=2):
    """Одиночное дифференцируемое распространение (сигнатура test.py)."""
    squeeze = phi.dim() == 2
    if squeeze:
        phi = phi.unsqueeze(0)
    device = phi.device
    key = (z, lam, dx, pad_factor, device)
    p = _single_prop_cache.get(key)
    if p is None:
        p = Propagator([z], lam=lam, dx=dx, N=phi.shape[-1], target_N=phi.shape[-1],
                       pad_factor=pad_factor)
        p._prepare(device)
        _single_prop_cache[key] = p
    K = p.propagate(phi)
    K = K[:, 0]
    return K.squeeze(0) if squeeze else K


# ────────────────────────────────────────────────────────────
# Метрики
# ────────────────────────────────────────────────────────────
def pearson_vec(a, b, eps=1e-8):
    a = a - a.mean(dim=(-2, -1), keepdim=True)
    b = b - b.mean(dim=(-2, -1), keepdim=True)
    num = (a * b).sum(dim=(-2, -1))
    den = a.norm(dim=(-2, -1)) * b.norm(dim=(-2, -1)) + eps
    return num / den


def ssim_simple_vec(a, b, eps=1e-8):
    mu_a = a.mean(dim=(-2, -1), keepdim=True)
    mu_b = b.mean(dim=(-2, -1), keepdim=True)
    var_a = a.var(dim=(-2, -1), unbiased=False, keepdim=True)
    var_b = b.var(dim=(-2, -1), unbiased=False, keepdim=True)
    cov = ((a - mu_a) * (b - mu_b)).mean(dim=(-2, -1), keepdim=True)
    C1, C2 = 0.01**2, 0.03**2
    ssim_map = ((2*mu_a*mu_b + C1) * (2*cov + C2)) / \
               ((mu_a**2 + mu_b**2 + C1) * (var_a + var_b + C2))
    return ssim_map.mean(dim=(-2, -1))


def pearson(a, b, eps=1e-8):
    return pearson_vec(a, b, eps).mean()


def ssim_simple(a, b, eps=1e-8):
    return ssim_simple_vec(a, b, eps).mean()


def tv_loss(phi):
    """TV-регуляризация фазы: сглаживает фазу и подавляет спекл."""
    dx = phi[..., 1:, :] - phi[..., :-1, :]
    dy = phi[..., :, 1:] - phi[..., :, :-1]
    return (dx ** 2).mean() + (dy ** 2).mean()


def normalize_intensity(I, target):
    """Приводит интенсивность I к mean/std мишени (как в лоссе)."""
    I_mean = I.mean(dim=(-2, -1), keepdim=True)
    I_std = I.std(dim=(-2, -1), keepdim=True) + 1e-8
    t_mean = target.mean(dim=(-2, -1), keepdim=True)
    t_std = target.std(dim=(-2, -1), keepdim=True) + 1e-8
    return torch.clamp((I - I_mean) / I_std * t_std + t_mean, 0, 1)


def total_loss(phi, targets, prop=None, tv_weight=0.0, loss_type="pearson"):
    """
    targets: список тензоров (старая сигнатура) или (Z, N, N) тензор.
    prop: опциональный Propagator (иначе используется кэшированный).

    loss_type:
        "pearson"       — 1 - КК (лучший для задачи, напрямую оптимизирует КК);
        "pearson+ssim"  — (1-КК) + (1-ИСС) (исходная формула);
        "mse"           — MSE нормализованной интенсивности.
    """
    if isinstance(targets, (list, tuple)):
        targets = torch.stack([t.to(phi.device) for t in targets])
    if prop is None:
        prop = _get_default_prop(phi.device)

    K = prop.propagate(phi)
    I = K.abs() ** 2
    I_norm = normalize_intensity(I, targets)

    if loss_type == "pearson":
        loss = (1 - pearson_vec(I_norm, targets)).mean()
    elif loss_type == "pearson+ssim":
        l = (1 - pearson_vec(I_norm, targets)) + (1 - ssim_simple_vec(I_norm, targets))
        loss = l.mean() + 0.5 * l.std()
    else:  # mse
        loss = ((I_norm - targets) ** 2).mean()

    if tv_weight > 0:
        loss = loss + tv_weight * tv_loss(phi)
    return loss


# ────────────────────────────────────────────────────────────
# Сети
# ────────────────────────────────────────────────────────────
class ConvBlock(nn.Module):
    """Двойная свёртка 3×3 + BN + ReLU (для U-Net)."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DoeUNet(nn.Module):
    """U-Net: на вход 4×128×128, на выход — фаза ДОЭ 128×128."""
    def __init__(self, width=1):
        super().__init__()
        w = width
        # энкодер (вход 4×128×128)
        self.enc1 = ConvBlock(4, 32 * w)       # 128×128
        self.enc2 = ConvBlock(32 * w, 64 * w)  # 64×64
        self.enc3 = ConvBlock(64 * w, 128 * w) # 32×32
        self.pool = nn.MaxPool2d(2)

        # декодер (32 → 64 → 128)
        self.up3 = nn.ConvTranspose2d(128 * w, 64 * w, 2, stride=2)
        self.dec3 = ConvBlock(128 * w, 64 * w)
        self.up2 = nn.ConvTranspose2d(64 * w, 32 * w, 2, stride=2)
        self.dec2 = ConvBlock(64 * w, 32 * w)

        self.out = nn.Conv2d(32 * w, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        d3 = self.dec3(torch.cat([self.up3(e3), e2], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e1], dim=1))

        phi = self.out(d2)
        return torch.clamp(phi, -torch.pi, torch.pi)


class DoeNet(nn.Module):
    """Базовая CNN: 5 свёрточных слоёв (128×128)."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4, 32, 3, padding=1),   nn.BatchNorm2d(32),  nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),  nn.BatchNorm2d(64),  nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=1), nn.BatchNorm2d(64),  nn.ReLU(),
            nn.Conv2d(64, 32, 3, padding=1),  nn.BatchNorm2d(32),  nn.ReLU(),
            nn.Conv2d(32, 1, 3, padding=1),
        )

    def forward(self, x):
        phi = self.net(x)
        return torch.clamp(phi, -torch.pi, torch.pi)


def make_model(arch, width=1):
    if arch == "unet":
        return DoeUNet(width=width)
    return DoeNet()


# ────────────────────────────────────────────────────────────
# Данные
# ────────────────────────────────────────────────────────────
def load_targets(dir_=TARGET_DIR):
    paths = [os.path.join(dir_, f"target_{i}.png") for i in range(4)]
    for p in paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Нет файла {p}. Сначала запустите prepare.py")
    return [np.asarray(Image.open(p), dtype=np.float32) / 255.0 for p in paths]


# ────────────────────────────────────────────────────────────
# Обучение
# ────────────────────────────────────────────────────────────
def train(epochs=500, lr=1e-3, log_every=25, arch="unet",
          tv_weight=0.0, grad_clip=10.0, compile_model=False, sched="onecycle",
          loss_type="pearson", width=1):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Устройство: {device}")
    print(f"Архитектура: {arch} (width×{width}), ДОЭ: {N}×{N}, TV={tv_weight}, "
          f"loss={loss_type}, шаг={DX*1e6:.1f} мкм")

    if device == "cuda":
        torch.backends.cudnn.benchmark = True

    targets_np = load_targets()
    targets_stack = torch.stack(
        [torch.tensor(t, device=device) for t in targets_np]
    )
    x_in = targets_stack.unsqueeze(0)                    # (1, 4, 128, 128)

    prop = Propagator(Z_LIST)
    prop._prepare(device)

    model = make_model(arch, width).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Параметров: {n_params:,}")

    if compile_model:
        model = torch.compile(model)
        print("torch.compile: включён")

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    if sched == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=epochs, eta_min=lr * 0.01
        )
    else:
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=lr, total_steps=epochs, pct_start=0.1
        )

    t0 = time.time()
    best_loss = float("inf")
    for epoch in range(1, epochs + 1):
        model.train()
        phi = model(x_in).squeeze()
        loss = total_loss(phi, targets_stack, prop, tv_weight=tv_weight,
                          loss_type=loss_type)

        opt.zero_grad()
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        sched.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            torch.save(model.state_dict(), "model_best.pth")

        if epoch % log_every == 0 or epoch == 1:
            with torch.no_grad():
                model.eval()
                phi_eval = model(x_in).squeeze()
                K = prop.propagate(phi_eval)
                I = K.abs() ** 2
                In = normalize_intensity(I, targets_stack)
                kks = pearson_vec(In, targets_stack)
            print(f"epoch {epoch:4d} | loss {loss.item():.4f} | "
                  f"КК(ср) {kks.mean().item():.4f} | {time.time()-t0:.1f}s")

    torch.save(model.state_dict(), "model.pth")
    print(f"\nОбучение завершено за {time.time()-t0:.1f}s")
    print(f"model.pth сохранён (best loss = {best_loss:.4f})")
    return model, targets_np, prop


# ────────────────────────────────────────────────────────────
# Сохранение результатов
# ────────────────────────────────────────────────────────────
def save_results(model, targets_np, prop=None):
    device = next(model.parameters()).device
    model.eval()

    targets_t = [torch.tensor(t, device=device) for t in targets_np]
    targets_stack = torch.stack(targets_t)
    x_in = targets_stack.unsqueeze(0)

    with torch.no_grad():
        phi = model(x_in).squeeze()

    if prop is None:
        prop = Propagator(Z_LIST)
        prop._prepare(device)

    # 1. ДОЭ: 0 ↔ -π, 255 ↔ +π (по условию)
    doe = ((phi.cpu().numpy() + np.pi) / (2 * np.pi) * 255).round().astype(np.uint8)
    Image.fromarray(doe).save("doe.png")
    print(f"Сохранено: doe.png ({phi.shape[0]}×{phi.shape[1]})")

    # 2. 4 восстановления + метрики (честная нормализация mean/std)
    rows = []
    K = prop.propagate(phi)               # (Z, 128, 128)
    I = K.abs() ** 2
    In = normalize_intensity(I, targets_stack)

    for i, z in enumerate(Z_LIST):
        In_i = In[i].cpu().numpy()
        Image.fromarray((np.clip(In_i, 0, 1) * 255).astype(np.uint8)).save(f"recovered_{i}.png")

        t = targets_np[i]
        kk = pearsonr(t.ravel(), In_i.ravel())[0]
        iss = ssim_np(t, In_i, data_range=1.0)
        rows.append((i, z * 1e3, kk, iss))
        print(f"recovered_{i}.png  (z={z*1e3:.1f} мм)  "
              f"КК={kk:.4f}  ИСС={iss:.4f}")

    # 3. metrics.txt
    with open("metrics.txt", "w") as f:
        f.write("img\tz_mm\tKK\tISS\n")
        for i, z_mm, kk, iss in rows:
            f.write(f"{i}\t{z_mm:.1f}\t{kk:.4f}\t{iss:.4f}\n")
    print("Сохранено: metrics.txt")

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
    ap.add_argument("--arch", choices=["unet", "cnn"], default="unet")
    ap.add_argument("--width", type=int, default=1,
                    help="множитель ширины U-Net (1/2/4)")
    ap.add_argument("--tv-weight", type=float, default=0.0,
                    help="вес TV-регуляризации фазы (0 = выключить)")
    ap.add_argument("--grad-clip", type=float, default=10.0,
                    help="клиппинг градиента (0 = выключить)")
    ap.add_argument("--sched", choices=["onecycle", "cosine"], default="onecycle")
    ap.add_argument("--loss", choices=["pearson", "pearson+ssim", "mse"],
                    default="pearson", help="функция потерь (pearson лучше всего)")
    ap.add_argument("--compile", action="store_true", help="включить torch.compile")
    ap.add_argument("--skip-train", action="store_true",
                    help="только сгенерировать результаты из model.pth")
    args = ap.parse_args()

    if args.skip_train:
        if not os.path.exists("model.pth"):
            print("Нет model.pth — сначала обучите.")
            return
        model = make_model(args.arch, args.width)
        model.load_state_dict(torch.load("model.pth", map_location="cpu"))
        targets_np = load_targets()
        save_results(model, targets_np)
        return

    model, targets_np, prop = train(args.epochs, args.lr, args.log_every,
                                    args.arch, args.tv_weight,
                                    args.grad_clip, args.compile, args.sched,
                                    args.loss, args.width)
    save_results(model, targets_np, prop)


if __name__ == "__main__":
    main()
