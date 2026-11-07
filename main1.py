from PIL import Image
from ai_cnn import DoeNet
import numpy as np, os
import torch

def corecting_img(paths, out_dir="targets", N=128):
    os.makedirs(out_dir, exist_ok=True)
    for i, p in enumerate(paths):
        img = Image.open(p).convert("L")
        img.save(f"{out_dir}/target_{i}.png")   # уже 8 бит, grayscale, PNG
        print(f"target_{i}.png", img.size, img.mode)


def img_to_phi(path, N=128):
    img = Image.open(path).convert("L")
    image = np.asarray(img, dtype = np.float32)/255.0
    phi = image * 2 * np.pi - np.pi

    return phi

def synthesis_doe(phi, z, lam=532e-9, dx=10e-6):
    N = phi.shape[0]
    j = 1j
    k = 2 * np.pi / lam

    # (u,v) на ПВМС
    u = (np.arange(N) - N//2) * dx
    U, V = np.meshgrid(u, u, indexing='xy')

    # H = exp(j*phi), чирп-фаза, FFT
    H = np.exp(j * phi)
    chirp_in = np.exp(j * np.pi / (lam * z) * (U**2 + V**2))
    F = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(H * chirp_in)))

    # (x,y) в плоскости восстановления — задаются физикой FFT
    dx_out = lam * z / (N * dx)
    x = (np.arange(N) - N//2) * dx_out
    X, Y = np.meshgrid(x, x, indexing='xy')

    # предмножитель и выходная фаза линзы
    A = np.exp(j * k * z) / (j * lam * z)
    chirp_out = np.exp(j * np.pi / (lam * z) * (X**2 + Y**2))

    return A * chirp_out * F

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

inp = ["test_img/imresizer-resize-png1.png", "test_img/imresizer-resize-png2.png", "test_img/imresizer-resize-png3.png", "test_img/imresizer-resize-png4.png"]
z_mm = [24.1, 25.3, 26.5, 27.7]
z_list = [z * 1e-3 for z in z_mm]

phi = img_to_phi("test_img_out/target_0.png")
neuro = synthesis_doe(phi, z_list[0])

print(neuro)

import torch

# 1. форма выхода
model = DoeNet()
x = torch.randn(1, 4, 128, 128)
phi = model(x)
assert phi.shape == (1, 1, 128, 128)
print("shape ok:", phi.shape)

# 2. диапазон φ
assert phi.min() >= -torch.pi - 1e-5
assert phi.max() <=  torch.pi + 1e-5
print("range ok:", phi.min().item(), phi.max().item())

# 3. градиенты текут
phi.sum().backward()
print("grad ok:", model.net[0].weight.grad is not None)
