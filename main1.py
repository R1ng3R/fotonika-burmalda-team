from PIL import Image
import numpy as np, os

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

inp = ["test_img/imresizer-resize-png1.png", "test_img/imresizer-resize-png2.png", "test_img/imresizer-resize-png3.png", "test_img/imresizer-resize-png4.png"]

corecting_img(inp, "test_img_out")
