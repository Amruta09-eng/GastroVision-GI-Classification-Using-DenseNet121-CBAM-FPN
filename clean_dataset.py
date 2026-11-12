"""
Dataset Cleaning Script for Gastrovision 5-Class Dataset
=========================================================
Removes:
  1. Blurry images       (Laplacian variance threshold)
  2. Exact duplicates    (MD5 hash)
  3. Near-duplicates     (perceptual hash pHash, Hamming distance <= threshold)
  4. Corrupted / unreadable files

Then splits the cleaned data into stratified train / val / test sets.

GPU Acceleration:
  - Tries CuPy (CUDA 12.x) first for GPU Laplacian convolution
  - Falls back to multi-threaded CPU (concurrent.futures + OpenCV)
  
Both GPU and CPU paths are maximally parallelized.
"""

import os, sys, shutil, hashlib, logging, argparse, random
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing

import numpy as np
from PIL import Image
import imagehash
import cv2
from tqdm import tqdm

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("cleaning_log.txt", mode="w"),
    ],
)
log = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────
ROOT         = Path(r"c:\Users\AMRUTA M PATIL\Downloads\Gastrovision_5classes")
OUTPUT_ROOT  = ROOT / "dataset"        # inside dataset folder (sandbox-safe)
TRASH_ROOT   = ROOT / "_removed"       # inside dataset folder (sandbox-safe)

BLUR_THRESHOLD  = 100.0   # Laplacian variance below this → blurry
PHASH_THRESHOLD = 8        # Hamming distance for near-duplicate detection
TRAIN_RATIO     = 0.70
VAL_RATIO       = 0.15
TEST_RATIO      = 0.15

IMG_EXTENSIONS  = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
SEED            = 42
NUM_WORKERS     = min(multiprocessing.cpu_count(), 8)   # CPU threads


# ═══════════════════════════════════════════════════════════════════════════════
#  GPU setup — CuPy (CUDA) if available, else multi-threaded CPU
# ═══════════════════════════════════════════════════════════════════════════════
USE_GPU = False
cp      = None

try:
    import cupy as _cp
    _cp.cuda.runtime.getDeviceCount()   # raises if no CUDA
    _test = _cp.array([1.0, 2.0])       # test allocation
    cp      = _cp
    USE_GPU = True
    gpu_info = f"CuPy CUDA GPU: {_cp.cuda.Device(0).name()}"
except Exception as _gpu_err:
    gpu_info = f"No CUDA GPU (reason: {_gpu_err}). Using {NUM_WORKERS}-thread CPU."

log.info(f"{'GPU' if USE_GPU else 'CPU'} mode: {gpu_info}")

# 3×3 Laplacian kernel
_LAP_KERNEL_NP = np.array([[0,  1, 0],
                             [1, -4, 1],
                             [0,  1, 0]], dtype=np.float32)
if USE_GPU:
    _LAP_KERNEL_GPU = cp.asarray(_LAP_KERNEL_NP)


# ── Blur score functions ──────────────────────────────────────────────────────
def _blur_score_gpu(path: Path):
    """GPU blur score via CuPy + cupyx.scipy.ndimage.convolve."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return path, None
    img_gpu = cp.asarray(img.astype(np.float32))
    from cupyx.scipy.ndimage import convolve
    lap = convolve(img_gpu, _LAP_KERNEL_GPU)
    return path, float(cp.var(lap).get())


def _blur_score_cpu(path: Path):
    """CPU blur score via OpenCV."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return path, None
    return path, float(cv2.Laplacian(img, cv2.CV_64F).var())


def compute_blur_scores_batch(paths):
    """
    Compute blur scores for a list of paths.
    Uses GPU (CuPy) if available, otherwise multi-threaded CPU.
    Returns: list of (path, score_or_None)
    """
    if USE_GPU:
        # Sequential on GPU but each is fast (kernel launch is cheap)
        results = []
        for p in tqdm(paths, desc="  Blur (GPU)", leave=False, unit="img"):
            results.append(_blur_score_gpu(p))
        return results
    else:
        # Multi-threaded CPU
        results = {}
        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as ex:
            futures = {ex.submit(_blur_score_cpu, p): p for p in paths}
            for f in tqdm(as_completed(futures), total=len(futures),
                          desc=f"  Blur (CPU×{NUM_WORKERS})", leave=False, unit="img"):
                p, score = f.result()
                results[p] = score
        return [(p, results[p]) for p in paths]


# ═══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════════════
def md5_hash(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_phash(path: Path):
    try:
        with Image.open(path) as img:
            return imagehash.phash(img)
    except Exception:
        return None


def move_to_trash(src: Path, reason: str):
    rel = src.relative_to(ROOT)
    dst = TRASH_ROOT / reason / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))


# ═══════════════════════════════════════════════════════════════════════════════
#  Main Cleaning Pipeline
# ═══════════════════════════════════════════════════════════════════════════════
def clean_dataset(blur_thresh: float, phash_thresh: int):
    random.seed(SEED)
    np.random.seed(SEED)

    # ── Discover all images ───────────────────────────────────────────────────
    all_images = defaultdict(list)
    SKIP_DIRS  = {"dataset", "_removed", "Problematic_Images_Quarantine"}    # skip our own output dirs
    for cls_dir in sorted(ROOT.iterdir()):
        if not cls_dir.is_dir():
            continue
        if cls_dir.name in SKIP_DIRS or cls_dir.name.startswith("_"):
            continue
        for f in cls_dir.iterdir():
            if f.suffix.lower() in IMG_EXTENSIONS:
                all_images[cls_dir.name].append(f)

    total_start = sum(len(v) for v in all_images.values())
    log.info(f"\nFound {total_start} images across {len(all_images)} classes")
    for cls, paths in all_images.items():
        log.info(f"  {cls}: {len(paths)} images")

    removed = defaultdict(int)
    kept    = defaultdict(list)

    for cls_name, paths in all_images.items():
        log.info(f"\n{'─'*60}")
        log.info(f"Processing class: '{cls_name}'  ({len(paths)} images)")

        # ── Step 1: Corruption check ─────────────────────────────────────────
        def check_corrupt(p):
            try:
                with Image.open(p) as img:
                    img.verify()
                return p, False
            except Exception:
                return p, True

        valid_paths = []
        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as ex:
            futures = {ex.submit(check_corrupt, p): p for p in paths}
            for f in tqdm(as_completed(futures), total=len(futures),
                          desc="  [1/4] Corruption check", leave=False):
                p, is_corrupt = f.result()
                if is_corrupt:
                    log.warning(f"    CORRUPTED → {p.name}")
                    move_to_trash(p, "corrupted")
                    removed["corrupted"] += 1
                else:
                    valid_paths.append(p)
        log.info(f"  Corruption: {len(paths)-len(valid_paths)} removed → {len(valid_paths)} remain")

        # ── Step 2: Blur detection ───────────────────────────────────────────
        log.info(f"  Blur threshold: {blur_thresh}  |  Mode: {'GPU' if USE_GPU else f'CPU×{NUM_WORKERS}'}")
        blur_results = compute_blur_scores_batch(valid_paths)
        non_blurry   = []
        for p, score in blur_results:
            if score is None:
                log.warning(f"    UNREADABLE → {p.name}")
                move_to_trash(p, "unreadable")
                removed["unreadable"] += 1
            elif score < blur_thresh:
                log.debug(f"    BLURRY (score={score:.2f}) → {p.name}")
                move_to_trash(p, "blurry")
                removed["blurry"] += 1
            else:
                non_blurry.append(p)
        log.info(f"  Blur: {len(valid_paths)-len(non_blurry)} removed → {len(non_blurry)} remain")

        # ── Step 3: Exact duplicates (MD5) ───────────────────────────────────
        seen_md5 = {}
        no_exact = []
        md5_results = {}
        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as ex:
            futures = {ex.submit(md5_hash, p): p for p in non_blurry}
            for f in tqdm(as_completed(futures), total=len(futures),
                          desc="  [3/4] Exact dup (MD5)", leave=False):
                p_path = futures[f]
                md5_results[p_path] = f.result()

        for p in non_blurry:   # preserve original order for determinism
            h = md5_results[p]
            if h in seen_md5:
                log.debug(f"    EXACT DUP → {p.name}")
                move_to_trash(p, "exact_duplicate")
                removed["exact_duplicate"] += 1
            else:
                seen_md5[h] = p
                no_exact.append(p)
        log.info(f"  Exact dup: {len(non_blurry)-len(no_exact)} removed → {len(no_exact)} remain")

        # ── Step 4: Near-duplicates (pHash) ──────────────────────────────────
        log.info(f"  pHash Hamming threshold: {phash_thresh}")
        phash_results = {}
        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as ex:
            futures = {ex.submit(compute_phash, p): p for p in no_exact}
            for f in tqdm(as_completed(futures), total=len(futures),
                          desc="  [4/4] Near dup (pHash)", leave=False):
                p_path = futures[f]
                phash_results[p_path] = f.result()

        hashes = [(p, phash_results[p]) for p in no_exact if phash_results[p] is not None]

        # Greedy deduplication: keep first occurrence, remove near-dups of it
        to_remove = set()
        for i in range(len(hashes)):
            if hashes[i][0] in to_remove:
                continue
            for j in range(i + 1, len(hashes)):
                if hashes[j][0] in to_remove:
                    continue
                if (hashes[i][1] - hashes[j][1]) <= phash_thresh:
                    to_remove.add(hashes[j][0])

        no_near = []
        for p, _ in hashes:
            if p in to_remove:
                log.debug(f"    NEAR DUP → {p.name}")
                move_to_trash(p, "near_duplicate")
                removed["near_duplicate"] += 1
            else:
                no_near.append(p)
        log.info(f"  Near dup: {len(no_exact)-len(no_near)} removed → {len(no_near)} remain")

        kept[cls_name] = no_near

    # ── Final summary ─────────────────────────────────────────────────────────
    total_kept = sum(len(v) for v in kept.values())
    log.info(f"\n{'═'*60}")
    log.info("REMOVAL SUMMARY")
    log.info(f"{'═'*60}")
    for reason, count in removed.items():
        log.info(f"  {reason:22s}: {count:5d} removed")
    log.info(f"  {'─'*30}")
    log.info(f"  {'TOTAL REMOVED':22s}: {total_start - total_kept:5d}  ({(total_start-total_kept)/total_start*100:.1f}%)")
    log.info(f"  {'TOTAL KEPT':22s}: {total_kept:5d}  / {total_start}")
    log.info("\n  Per-class (after cleaning):")
    for cls, paths in kept.items():
        log.info(f"    {cls}: {len(paths)}")

    return kept


# ═══════════════════════════════════════════════════════════════════════════════
#  Stratified Train / Val / Test Split + Copy
# ═══════════════════════════════════════════════════════════════════════════════
def split_and_copy(kept: dict):
    assert abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) < 1e-6

    split_counts = defaultdict(lambda: defaultdict(int))

    for cls_name, paths in kept.items():
        random.shuffle(paths)
        n       = len(paths)
        n_train = int(n * TRAIN_RATIO)
        n_val   = int(n * VAL_RATIO)

        splits = {
            "train": paths[:n_train],
            "val"  : paths[n_train : n_train + n_val],
            "test" : paths[n_train + n_val :],
        }

        for split_name, split_paths in splits.items():
            dst_dir = OUTPUT_ROOT / split_name / cls_name
            dst_dir.mkdir(parents=True, exist_ok=True)

            def _copy(src):
                shutil.copy2(str(src), str(dst_dir / src.name))

            with ThreadPoolExecutor(max_workers=NUM_WORKERS) as ex:
                list(tqdm(ex.map(_copy, split_paths),
                          total=len(split_paths),
                          desc=f"  Copying {split_name}/{cls_name[:25]}",
                          leave=False))
            split_counts[split_name][cls_name] = len(split_paths)

    log.info(f"\n{'═'*60}")
    log.info(f"TRAIN / VAL / TEST SPLIT  →  {OUTPUT_ROOT}")
    log.info(f"{'═'*60}")
    for split in ["train", "val", "test"]:
        total = sum(split_counts[split].values())
        log.info(f"\n  [{split.upper()}]  {total} images")
        for cls, cnt in split_counts[split].items():
            log.info(f"    {cls}: {cnt}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Entry point
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean & split Gastrovision dataset")
    parser.add_argument("--blur_thresh", type=float, default=BLUR_THRESHOLD,
                        help=f"Laplacian variance threshold (default {BLUR_THRESHOLD}). "
                             "Lower = keep more; higher = stricter blur removal.")
    parser.add_argument("--phash_thresh", type=int, default=PHASH_THRESHOLD,
                        help=f"pHash Hamming distance (default {PHASH_THRESHOLD}). "
                             "0 = only identical perceptual copies; 10 = liberal.")
    parser.add_argument("--no_split", action="store_true",
                        help="Skip train/val/test split.")
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("  Gastrovision Dataset Cleaner")
    log.info(f"  Blur threshold  : {args.blur_thresh}")
    log.info(f"  pHash threshold : {args.phash_thresh}")
    log.info(f"  Acceleration    : {'GPU (CuPy CUDA)' if USE_GPU else f'CPU ({NUM_WORKERS} threads)'}")
    log.info(f"  Output dir      : {OUTPUT_ROOT}")
    log.info(f"  Trash dir       : {TRASH_ROOT}")
    log.info("=" * 60)

    kept = clean_dataset(args.blur_thresh, args.phash_thresh)

    if not args.no_split:
        split_and_copy(kept)

    log.info("\n✅  Done! Full log saved to: cleaning_log.txt")
