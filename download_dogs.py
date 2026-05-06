"""Download or split Stanford Dogs into train/evaluate/test."""
from __future__ import annotations

import argparse
import random
import shutil
import tarfile
from collections import defaultdict
from pathlib import Path

import requests


IMAGES_URL = "http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar"
ANNOT_URL = "http://vision.stanford.edu/aditya86/ImageNetDogs/annotation.tar"
IMAGE_EXTS = {'.jpg', '.jpeg', '.png'}


def download_file(url: str, dest_path: Path, chunk_size: int = 1024 * 1024):
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    print(f'Downloading {url} -> {dest_path}')
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(dest_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
    print('Downloaded.')


def safe_extract(tar_path: Path, dest_dir: Path):
    with tarfile.open(tar_path, 'r') as tar:
        tar.extractall(path=dest_dir)


def resolve_images_root(from_local: str | None, extract_dir: Path) -> Path:
    if from_local:
        root = Path(from_local)
        candidates = [root / 'Images', root]
        for candidate in candidates:
            if candidate.exists() and candidate.is_dir() and any(p.is_dir() for p in candidate.iterdir()):
                return candidate
        raise RuntimeError(f'Could not find Images folder inside {root}')

    img_root = extract_dir / 'Images'
    if not img_root.exists():
        raise RuntimeError(f'{img_root} not found')
    return img_root


def split_dataset(img_root: Path, out_root: Path, seed: int = 42, overwrite: bool = False):
    if out_root.exists() and overwrite:
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    classes = sorted([d for d in img_root.iterdir() if d.is_dir()])
    stats = {}

    for cls in classes:
        imgs = sorted([p for p in cls.iterdir() if p.suffix.lower() in IMAGE_EXTS])
        if not imgs:
            continue
        rng.shuffle(imgs)
        n = len(imgs)
        n_train = int(n * 0.70)
        n_eval = int(n * 0.15)
        n_test = n - n_train - n_eval
        parts = {
            'train': imgs[:n_train],
            'evaluate': imgs[n_train:n_train + n_eval],
            'test': imgs[n_train + n_eval:],
        }
        for part, files in parts.items():
            dst = out_root / part / 'images' / cls.name
            dst.mkdir(parents=True, exist_ok=True)
            for src in files:
                shutil.copy2(src, dst / src.name)
        stats[cls.name] = {k: len(v) for k, v in parts.items()}

    totals = defaultdict(int)
    for values in stats.values():
        for split, count in values.items():
            totals[split] += count

    print('Totals:', dict(totals))
    print('Classes:', len(stats))
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='./input', help='Output folder for split dataset')
    parser.add_argument('--from-local', dest='from_local', default=None,
                        help='Path to raw Stanford Dogs folder; may be the parent of Images or Images itself')
    parser.add_argument('--url-images', default=IMAGES_URL)
    parser.add_argument('--url-annotations', default=ANNOT_URL)
    parser.add_argument('--overwrite', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    out_root = Path(args.out)
    extract_dir = Path('./extracted')
    extract_dir.mkdir(parents=True, exist_ok=True)

    if args.from_local:
        img_root = resolve_images_root(args.from_local, extract_dir)
    else:
        downloads = Path('./downloads')
        downloads.mkdir(parents=True, exist_ok=True)
        images_tar = downloads / 'images.tar'
        ann_tar = downloads / 'annotation.tar'

        if not images_tar.exists():
            download_file(args.url_images, images_tar)
        if not ann_tar.exists():
            try:
                download_file(args.url_annotations, ann_tar)
            except Exception as exc:
                print(f'Warning: annotation download failed: {exc}')

        print('Extracting images...')
        safe_extract(images_tar, extract_dir)
        img_root = resolve_images_root(None, extract_dir)

    split_dataset(img_root, out_root, seed=args.seed, overwrite=args.overwrite)
    print(f'Split dataset saved to: {out_root.resolve()}')


if __name__ == '__main__':
    main()
