# build_dataset_from_composites_with_nose_masks.py
# Hard-coded pipeline: mixed (left|right) -> dataset/{train,val,test}/{input,target,mask_input}

import os, random, shutil
from pathlib import Path
from typing import List, Tuple, Optional
import cv2
import numpy as np
from tqdm import tqdm

import torch
import face_alignment  # pip install face-alignment opencv-python torch torchvision numpy tqdm

# =========================
# HARD-CODED SETTINGS
# =========================
SRC_MIXED = Path("/Users/atyantjain/Desktop/MDSI/Ilab Capstone/Data")           # <-- folder with your 1204 composite images (left|right in one file)
DST_ROOT  = Path("/Users/atyantjain/Desktop/MDSI/Ilab Capstone/Preprocess")        # <-- output dataset root to be CREATED
PAIR_ORDER = "LR"                              # "LR" = left is input (A), right is target (B); use "RL" to flip
SPLIT_RATIOS = (0.8, 0.1, 0.1)                 # train, val, test
SEED = 42

SAVE_OVERLAYS = True                           # save heatmap overlays for fast QA
SAVE_SOFT     = False                          # also save soft masks; binary goes to mask_input/

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# =========================
# Utilities
# =========================
def is_image(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMG_EXTS

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def list_images(folder: Path) -> List[Path]:
    if not folder.exists():
        raise FileNotFoundError(f"Source folder not found: {folder}")
    imgs = [p for p in folder.iterdir() if is_image(p)]
    imgs.sort()
    return imgs

def split_train_val_test(items: List[Path], ratios=(0.8,0.1,0.1), seed=42):
    assert abs(sum(ratios) - 1.0) < 1e-6
    random.Random(seed).shuffle(items)
    n = len(items)
    n_train = int(n*ratios[0])
    n_val   = int(n*ratios[1])
    train = items[:n_train]
    val   = items[n_train:n_train+n_val]
    test  = items[n_train+n_val:]
    return train, val, test

def save_jpg(path: Path, img_bgr: np.ndarray, q=95):
    ensure_dir(path.parent)
    cv2.imwrite(str(path), img_bgr, [cv2.IMWRITE_JPEG_QUALITY, q])

# =========================
# Split composite into (input, target)
# =========================
def split_composite_lr(img_bgr: np.ndarray, pair_order="LR") -> Tuple[np.ndarray, np.ndarray]:
    h, w = img_bgr.shape[:2]
    mid = w // 2
    left  = img_bgr[:, :mid].copy()
    right = img_bgr[:, mid:].copy()
    if pair_order.upper() == "LR":
        return left, right
    else:
        return right, left

# =========================
# Landmarks → rough polygon → GrabCut refine
# =========================
def build_nose_poly_from_68(pts68: np.ndarray) -> np.ndarray:
    # 0-based indices: bridge 27..30, alar 31..35
    bridge = [27, 28, 29, 30]
    alar   = [31, 32, 33, 34, 35]
    order  = bridge + alar + bridge[::-1][:1]
    poly   = np.round(pts68[order]).astype(np.int32)
    return poly

def refine_mask_grabcut(img_bgr: np.ndarray, init_mask_255: np.ndarray,
                        rect_margin: int = 16, iters: int = 3) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    init = (init_mask_255 > 0).astype(np.uint8)
    ys, xs = np.where(init > 0)
    if len(xs) == 0:
        return init_mask_255
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    x0 = max(0, x0 - rect_margin); y0 = max(0, y0 - rect_margin)
    x1 = min(w-1, x1 + rect_margin); y1 = min(h-1, y1 + rect_margin)

    GC_BGD, GC_FGD, GC_PR_BGD, GC_PR_FGD = 0,1,2,3
    gc_mask = np.full((h,w), GC_PR_BGD, np.uint8)
    gc_mask[y0:y1+1, x0:x1+1] = GC_PR_FGD

    sure_fg = cv2.dilate(init*255, np.ones((5,5), np.uint8), 1) > 0
    gc_mask[sure_fg] = GC_FGD

    roi = np.zeros((h,w), np.uint8); roi[y0:y1+1, x0:x1+1] = 1
    gc_mask[roi==0] = GC_BGD

    bgdModel = np.zeros((1,65), np.float64)
    fgdModel = np.zeros((1,65), np.float64)
    cv2.grabCut(img_bgr, gc_mask, None, bgdModel, fgdModel, iters, cv2.GC_INIT_WITH_MASK)
    mask_ref = np.where((gc_mask==GC_FGD)|(gc_mask==GC_PR_FGD), 255, 0).astype(np.uint8)
    mask_ref = cv2.morphologyEx(mask_ref, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), 1)
    return mask_ref

class NoseMasker:
    def __init__(self, device: Optional[str] = None):
        device = device or DEVICE
        self.fa = face_alignment.FaceAlignment(
            face_alignment.LandmarksType.TWO_D,
            device=device,
            flip_input=False
        )

    def make_soft_mask(self, img_bgr: np.ndarray, feather_px: int = 9) -> Optional[np.ndarray]:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        try:
            preds = self.fa.get_landmarks(img_rgb)
        except Exception:
            preds = None
        if not preds:
            return None
        pts = preds[0]
        poly = build_nose_poly_from_68(pts)
        rough = np.zeros(img_bgr.shape[:2], np.uint8)
        cv2.fillConvexPoly(rough, poly, 255)
        refined = refine_mask_grabcut(img_bgr, rough, rect_margin=16, iters=3)
        if feather_px > 0:
            k = feather_px | 1
            return cv2.GaussianBlur(refined, (k,k), 0)
        return refined

# =========================
# Build dataset
# =========================
def process_split(name: str, files: List[Path], dst_root: Path,
                  pair_order="LR", save_overlays=True, save_soft=False):
    # Create split dirs
    split_dir = dst_root / name
    input_dir = split_dir / "input"
    target_dir = split_dir / "target"
    mask_dir  = split_dir / "mask_input"
    soft_dir  = split_dir / "mask_soft"
    ovl_dir   = split_dir / "_mask_debug"
    for d in [input_dir, target_dir, mask_dir, (soft_dir if save_soft else None), (ovl_dir if save_overlays else None)]:
        if d: ensure_dir(d)

    masker = NoseMasker()
    ok = fail = 0

    for p in tqdm(files, desc=f"[{name}]"):
        bgr = cv2.imread(str(p))
        if bgr is None:
            fail += 1; continue

        # 1) split composite
        a_bgr, b_bgr = split_composite_lr(bgr, pair_order=pair_order)

        # 2) filenames
        stem = p.stem
        in_name = f"{stem}.jpg"       # keep one name per pair (for input/target/mask)
        tgt_name = f"{stem}.jpg"

        # 3) save input & target
        save_jpg(input_dir / in_name, a_bgr)
        save_jpg(target_dir / tgt_name, b_bgr)

        # 4) make mask for INPUT image only
        soft = masker.make_soft_mask(a_bgr, feather_px=9)
        if soft is None:
            # still save blank mask to keep dataset aligned
            blank = np.zeros(a_bgr.shape[:2], np.uint8)
            cv2.imwrite(str(mask_dir / in_name), blank)
            fail += 1
            continue

        hard = (soft > 127).astype(np.uint8) * 255
        cv2.imwrite(str(mask_dir / in_name), hard)
        if save_soft:
            cv2.imwrite(str(soft_dir / in_name), soft)

        if save_overlays:
            overlay = a_bgr.copy()
            colored = cv2.applyColorMap(soft, cv2.COLORMAP_JET)
            cv2.addWeighted(colored, 0.35, overlay, 0.65, 0, overlay)
            cv2.imwrite(str(ovl_dir / f"{stem}_overlay.jpg"), overlay)

        ok += 1

    print(f"[{name}] wrote input/target/mask for {ok} images; landmark/mask FAILS: {fail}")

def main():
    print(f"Device: {DEVICE}")
    print(f"Reading composites from: {SRC_MIXED}")
    print(f"Writing dataset to    : {DST_ROOT}")
    ensure_dir(DST_ROOT)

    files = list_images(SRC_MIXED)
    if len(files) == 0:
        raise RuntimeError(f"No images found in {SRC_MIXED}")

    train, val, test = split_train_val_test(files, SPLIT_RATIOS, SEED)
    print(f"Total: {len(files)} | train {len(train)}  val {len(val)}  test {len(test)}")

    process_split("train", train, DST_ROOT, pair_order=PAIR_ORDER,
                  save_overlays=SAVE_OVERLAYS, save_soft=SAVE_SOFT)
    process_split("val",   val,   DST_ROOT, pair_order=PAIR_ORDER,
                  save_overlays=SAVE_OVERLAYS, save_soft=SAVE_SOFT)
    process_split("test",  test,  DST_ROOT, pair_order=PAIR_ORDER,
                  save_overlays=SAVE_OVERLAYS, save_soft=SAVE_SOFT)

    print("DONE.")

if __name__ == "__main__":
    main()
