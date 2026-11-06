"""
Подготовка исходных изображений для обучения ДОЭ-сети.

Использование:
    python prepare.py my_images
    python prepare.py my_images --n 4 --size 128 --out targets
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def collect_images(src_dir: Path):
    if not src_dir.is_dir():
        raise NotADirectoryError(f"Не папка: {src_dir}")
    files = sorted(p for p in src_dir.iterdir()
                   if p.is_file() and p.suffix.lower() in EXTS)
    if not files:
        raise FileNotFoundError(f"В папке нет картинок: {src_dir}")
    return files


def prepare_targets(paths, out_dir="targets", N=128):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for i, p in enumerate(paths):
        img = Image.open(p).convert("L").resize((N, N), Image.LANCZOS)
        out_path = out_dir / f"target_{i}.png"
        img.save(out_path)
        saved.append(out_path)
        print(f"[{i}] {p.name:30s} → {out_path}  ({img.size}, {img.mode})")
    return saved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=str, help="папка с исходными картинками")
    ap.add_argument("--out", type=str, default="targets")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--size", type=int, default=128)
    args = ap.parse_args()

    try:
        files = collect_images(Path(args.src))
    except (NotADirectoryError, FileNotFoundError) as e:
        print(f"Ошибка: {e}", file=sys.stderr); sys.exit(1)

    if len(files) < args.n:
        print(f"Внимание: картинок {len(files)}, нужно {args.n}. Возьму все.",
              file=sys.stderr)
        selected = files
    else:
        selected = files[: args.n]

    print(f"Найдено {len(files)}, выбрано {len(selected)}.\n")
    saved = prepare_targets(selected, out_dir=args.out, N=args.size)

    # проверка
    ok = True
    for p in saved:
        im = Image.open(p)
        ok &= (im.size == (args.size, args.size)) and (im.mode == "L")
    print(f"\nГотово: {len(saved)} файлов в '{args.out}/' "
          f"{'✅' if ok else '❌ (проверьте вывод)'}")
    if ok:
        print("Теперь можно запускать: python train.py")


if __name__ == "__main__":
    main()
