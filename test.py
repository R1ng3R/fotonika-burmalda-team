"""
Диагностика пайплайна синтеза ДОЭ.

Использование:
    python test.py                           # полная диагностика (без переобучения)
    python test.py --retrain-1               # + переобучение на 1 картинке (30 сек)
    python test.py --retrain-1 --arch unet   # то же, но U-Net
"""

import argparse
import os
import numpy as np
import torch
from PIL import Image
from scipy.stats import pearsonr

from train import (
    LAM, DX, N, Z_LIST, TARGET_DIR,
    propagate, propagate_torch,
    total_loss, pearson, ssim_simple,
    DoeNet, DoeUNet, load_targets,
)

import torch
from train import propagate_torch

phi = torch.randn(128, 128, requires_grad=True)
K = propagate_torch(phi, 24.1e-3)
(K.abs()**2).sum().backward()
print(f"grad is None: {phi.grad is None}")
print(f"grad mean: {phi.grad.abs().mean():.6f}")

def sep(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


# ────────────────────────────────────────────────────────────
# 1. Данные
# ────────────────────────────────────────────────────────────
def check_data():
    sep("1. ДАННЫЕ (targets/)")
    if not os.path.isdir(TARGET_DIR):
        print(f"❌ Нет папки {TARGET_DIR}/ — запустите prepare.py")
        return False

    targets = load_targets()
    ok = True
    for i, t in enumerate(targets):
        std = t.std()
        mean = t.mean()
        status = "✅" if std > 0.1 else "❌ (однотонная!)"
        print(f"target_{i}: shape={t.shape}, mean={mean:.3f}, "
              f"std={std:.3f}  {status}")
        if std < 0.1:
            ok = False
    return ok


# ────────────────────────────────────────────────────────────
# 2. Физика: propagate vs propagate_torch
# ────────────────────────────────────────────────────────────
def check_physics():
    sep("2. ФИЗИКА: propagate vs propagate_torch")
    phi_np = np.random.uniform(-np.pi, np.pi, (N, N)).astype(np.float64)

    for z in Z_LIST:
        K_np = propagate(phi_np, z)
        K_t  = propagate_torch(torch.tensor(phi_np), z).numpy()
        rel = np.abs(K_np - K_t).max() / (np.abs(K_np).max() + 1e-12)
        status = "✅" if rel < 1e-3 else "❌ (расхождение!)"
        print(f"z={z*1e3:.1f} мм: rel = {rel:.2e}  {status}")


# ────────────────────────────────────────────────────────────
# 3. Физика: зависит ли от z
# ────────────────────────────────────────────────────────────
def check_z_dependence():
    sep("3. ФИЗИКА: зависит ли propagate от z?")
    phi = np.random.uniform(-np.pi, np.pi, (N, N))
    results = [np.abs(propagate(phi, z))**2 for z in Z_LIST]
    results = [r / r.max() for r in results]

    ok = True
    for i in range(4):
        for j in range(i + 1, 4):
            diff = np.abs(results[i] - results[j]).mean()
            status = "✅" if diff > 0.01 else "❌ (не зависит от z!)"
            print(f"z{i} vs z{j}: mean|diff| = {diff:.4f}  {status}")
            if diff < 0.01:
                ok = False
    return ok


# ────────────────────────────────────────────────────────────
# 4. chirp_out — без padding и с padding
# ────────────────────────────────────────────────────────────
def check_chirp():
    sep("4. CHIRP_OUT: скорость осцилляции (без padding)")
    print(f"{'z, мм':>8} | {'phase max':>12} | {'период, px':>12} | статус")
    print("-" * 60)
    for z in Z_LIST:
        dx_out = LAM * z / (N * DX)
        x = (np.arange(N) - N // 2) * dx_out
        X, Y = np.meshgrid(x, x, indexing="xy")
        phase = np.pi / (LAM * z) * (X**2 + Y**2)
        center_idx = N // 2
        d_phase = np.gradient(phase[center_idx, :]).max()
        period = 2 * np.pi / (d_phase + 1e-12)
        status = "✅" if period > 2 else "⚠️ (быстро, нужен padding)"
        print(f"{z*1e3:8.1f} | {phase.max():12.1f} | {period:12.2f} | {status}")


def check_chirp_padded(pad_factor=2):
    sep(f"4b. CHIRP_OUT С PADDING (×{pad_factor})")
    N_pad = N * pad_factor
    print(f"{'z, мм':>8} | {'период, px':>12} | статус")
    print("-" * 40)
    for z in Z_LIST:
        dx_out = LAM * z / (N_pad * DX)
        x = (np.arange(N_pad) - N_pad // 2) * dx_out
        X, Y = np.meshgrid(x, x, indexing="xy")
        phase = np.pi / (LAM * z) * (X**2 + Y**2)
        center = N_pad // 2
        d_phase = np.gradient(phase[center, :]).max()
        period = 2 * np.pi / (d_phase + 1e-12)
        status = "✅" if period > 2 else "⚠️"
        print(f"{z*1e3:8.1f} | {period:12.2f} | {status}")


# ────────────────────────────────────────────────────────────
# 5. doe.png — не однотонный ли
# ────────────────────────────────────────────────────────────
def check_doe():
    sep("5. doe.png: не однотонный ли?")
    if not os.path.exists("doe.png"):
        print("⚠️ Нет doe.png — сначала обучите модель (это не ошибка, "
              "если вы ещё не обучали)")
        return None

    doe = np.asarray(Image.open("doe.png"))
    mean = doe.mean()
    std = doe.std()

    print(f"shape: {doe.shape}")
    print(f"dtype: {doe.dtype}")
    print(f"min/max: {doe.min()} / {doe.max()}")
    print(f"mean: {mean:.1f} (норма ~127)")
    print(f"std: {std:.1f} (норма > 30)")

    ok = True
    if std < 10:
        print("❌ doe.png почти однотонный — сеть выучила константу!")
        ok = False
    elif abs(mean - 127) > 20:
        print("⚠️ mean сильно отличается от 127")
        ok = False
    else:
        print("✅ doe.png выглядит как фаза")
    return ok


# ────────────────────────────────────────────────────────────
# 6. recovered — КК и MSE
# ────────────────────────────────────────────────────────────
def check_recovered():
    sep("6. RECOVERED: КК, MSE, ИСС (если есть recovered_*.png)")
    targets = load_targets()

    any_found = False
    print(f"{'img':>4} | {'z, мм':>6} | {'КК':>6} | {'MSE':>7} | {'ИСС':>6}")
    print("-" * 45)

    for i, z in enumerate(Z_LIST):
        r_path = f"recovered_{i}.png"
        if not os.path.exists(r_path):
            print(f"{i:>4} | {z*1e3:6.1f} |   —   (нет файла)")
            continue

        any_found = True
        t = targets[i]
        r = np.asarray(Image.open(r_path), np.float32) / 255

        kk = pearsonr(t.ravel(), r.ravel())[0]
        mse = ((t - r) ** 2).mean()
        iss = ssim_simple(torch.tensor(t), torch.tensor(r)).item()

        print(f"{i:>4} | {z*1e3:6.1f} | {kk:6.3f} | {mse:7.4f} | {iss:6.3f}")

    if not any_found:
        print("\n⚠️ Нет recovered_*.png — сначала обучите модель")


# ────────────────────────────────────────────────────────────
# 7. Переобучение на 1 картинке (опционально)
# ────────────────────────────────────────────────────────────
def retrain_one(epochs=300, lr=1e-3, arch="cnn"):
    sep(f"7. ПЕРЕОБУЧЕНИЕ НА 1 КАРТИНКЕ ({arch}, {epochs} эпох)")
    print("Если КК > 0.9 — пайплайн работает, проблема в 4 картинках")
    print("Если КК < 0.5 — пайплайн сломан (физика / loss)")

    t = np.asarray(Image.open(f"{TARGET_DIR}/target_0.png"),
                   np.float32) / 255
    targets = [t, t, t, t]   # 4 копии одной картинки
    targets_t = [torch.tensor(x) for x in targets]
    x_in = torch.stack(targets_t).unsqueeze(0)

    # ВАЖНО: создаём модель с нуля, НЕ грузим model.pth
    model = (DoeUNet() if arch == "unet" else DoeNet())
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Параметров: {n_params:,}")

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    for epoch in range(1, epochs + 1):
        model.train()
        phi = model(x_in).squeeze()
        loss = total_loss(phi, targets_t)
        opt.zero_grad(); loss.backward(); opt.step()
        sched.step()

        if epoch % 50 == 0 or epoch == 1:
            with torch.no_grad():
                model.eval()
                phi_eval = model(x_in).squeeze()
                kks = []
                for tt, z in zip(targets_t, Z_LIST):
                    K = propagate_torch(phi_eval, z)
                    I = K.abs()**2
                    I = I / (I.max() + 1e-8)
                    kks.append(pearson(I, tt).item())
                kk_mean = float(np.mean(kks))
            print(f"epoch {epoch:4d} | loss {loss.item():.4f} | "
                  f"КК(ср) {kk_mean:.4f}")

    with torch.no_grad():
        model.eval()
        phi = model(x_in).squeeze()
        kks = []
        for tt, z in zip(targets_t, Z_LIST):
            K = propagate_torch(phi, z)
            I = K.abs()**2
            I = I / (I.max() + 1e-8)
            kks.append(pearson(I, tt).item())

    print(f"\nИТОГ: КК = {[f'{k:.3f}' for k in kks]}")
    if np.mean(kks) > 0.9:
        print("✅ Пайплайн РАБОТАЕТ — проблема в 4 разных картинках")
    else:
        print("❌ Пайплайн СЛОМАН — проблема в физике / loss")
    return np.mean(kks)


# ────────────────────────────────────────────────────────────
# main
# ────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrain-1", action="store_true",
                    help="+ переобучение на 1 картинке (30 сек)")
    ap.add_argument("--arch", choices=["cnn", "unet"], default="cnn")
    ap.add_argument("--epochs", type=int, default=300)
    args = ap.parse_args()

    print("\n" + "=" * 60)
    print("ДИАГНОСТИКА ПАЙПЛАЙНА СИНТЕЗА ДОЭ")
    print("=" * 60)

    data_ok = check_data()
    check_physics()
    z_ok = check_z_dependence()
    check_chirp()
    check_chirp_padded()
    doe_ok = check_doe()
    check_recovered()

    sep("РЕЗЮМЕ")
    print(f"Данные:       {'✅' if data_ok else '❌'}")
    print(f"Зависит от z: {'✅' if z_ok else '❌'}")
    print(f"doe.png:      {'✅' if doe_ok else ('⚠️ нет файла' if doe_ok is None else '❌')}")

    if args.retrain_1:
        retrain_one(args.epochs, 1e-3, args.arch)
    else:
        print("\nДля проверки «может ли сеть выучить 1 картинку»:")
        print(f"  python test.py --retrain-1 --arch {args.arch}")


if __name__ == "__main__":
    main()
