#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt


# -----------------------
# Config
# -----------------------
CONFIG = {
    "IMG_SIZE": (512, 512),
    "NUM_WORKERS": 4,
    "ENCODER": "efficientnet-b0",
    "WEIGHTS": "imagenet",
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
}

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# -----------------------
# Utils
# -----------------------
def denormalize(tensor_chw: torch.Tensor) -> np.ndarray:
    """CHW normalized tensor -> HWC RGB float [0,1]"""
    img = tensor_chw.permute(1, 2, 0).detach().cpu().numpy()
    img = img * STD + MEAN
    return np.clip(img, 0, 1)


def list_images(img_dir: Path, recursive: bool) -> List[Path]:
    exts = {".png", ".jpg", ".jpeg"}
    if recursive:
        return sorted([p for p in img_dir.rglob("*") if p.is_file() and p.suffix.lower() in exts])
    return sorted([p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in exts])


def read_both_rg01(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    both heatmap PNG (BGR uint8):
      G(channel 1) = disc
      R(channel 2) = macula
      B(channel 0) = unused
    return:
      disc01, mac01 in float32 [0,1], shape (H,W)
    """
    hm = cv2.imread(str(path), cv2.IMREAD_COLOR)  # BGR
    if hm is None:
        raise FileNotFoundError(f"Failed to read BOTH heatmap: {path}")

    disc01 = hm[..., 1].astype(np.float32) / 255.0  # G → disc
    mac01  = hm[..., 2].astype(np.float32) / 255.0  # R → macula
    return disc01, mac01


def calculate_metrics_binary(pred01: torch.Tensor, tgt01: torch.Tensor, threshold: float) -> Tuple[float, float]:
    pred = (pred01 > threshold).float()
    tgt  = (tgt01  > threshold).float()

    inter = (pred * tgt).sum()
    union = pred.sum() + tgt.sum() - inter
    dice_den = pred.sum() + tgt.sum()

    iou  = (inter / union) if union > 0 else torch.tensor(0.0, device=pred.device)
    dice = (2 * inter / dice_den) if dice_den > 0 else torch.tensor(0.0, device=pred.device)
    return float(iou.item()), float(dice.item())


def calculate_micro_metrics(pred2: torch.Tensor, tgt2: torch.Tensor, threshold: float) -> Tuple[float, float]:
    """pred2/tgt2: (2,H,W)"""
    pred = (pred2 > threshold).float()
    tgt  = (tgt2  > threshold).float()

    inter = (pred * tgt).sum()
    union = pred.sum() + tgt.sum() - inter
    dice_den = pred.sum() + tgt.sum()

    iou  = (inter / union) if union > 0 else torch.tensor(0.0, device=pred.device)
    dice = (2 * inter / dice_den) if dice_den > 0 else torch.tensor(0.0, device=pred.device)
    return float(iou.item()), float(dice.item())


def overlay_two_heatmaps_rgb(
    orig_rgb01: np.ndarray,
    disc01: np.ndarray,
    mac01: np.ndarray,
    alpha: float = 0.45,
    disc_cmap: int = cv2.COLORMAP_JET,
    mac_cmap: int = cv2.COLORMAP_HOT,
) -> np.ndarray:
    """
    debug용 overlay: 두 heatmap을 colormap으로 만든 뒤 섞어서 원본에 overlay.
    return: RGB uint8
    """
    base_u8 = (orig_rgb01 * 255.0).astype(np.uint8)
    base_bgr = cv2.cvtColor(base_u8, cv2.COLOR_RGB2BGR)

    d_u8 = np.clip(disc01 * 255.0, 0, 255).astype(np.uint8)
    m_u8 = np.clip(mac01 * 255.0, 0, 255).astype(np.uint8)

    d_col = cv2.applyColorMap(d_u8, disc_cmap)  # BGR
    m_col = cv2.applyColorMap(m_u8, mac_cmap)   # BGR
    hm_mix = cv2.addWeighted(d_col, 0.5, m_col, 0.5, 0.0)

    out_bgr = cv2.addWeighted(base_bgr, 1.0 - alpha, hm_mix, alpha, 0.0)
    return cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)


# -----------------------
# Pairing (BOTH GT PATH)
# -----------------------
def build_pairs_for_split_both(
    det_root: Path,
    split: str,
    recursive_images: bool,
    test_case: str,
) -> Tuple[List[Path], List[Path]]:
    """
    imgaes: images/{split}/... 
    GT: det_root/heatmap/heatmap/{test_case}_{split}/{filename}.png
    """
    img_dir = det_root / "images" / split
    if not img_dir.is_dir():
        raise FileNotFoundError(f"images/{split} not found: {img_dir}")

    run_tag = f"{test_case}_{split}"
    both_gt_dir = det_root / "heatmap" / "heatmap" / run_tag
    if not both_gt_dir.is_dir():
        raise FileNotFoundError(
            f"BOTH GT dir not found.\n"
            f"dir: {both_gt_dir}\n"
            f"(expected: det_root/heatmap/heatmap/{test_case}_{split}/)"
        )

    img_paths_all = list_images(img_dir, recursive=recursive_images)

    seen: Dict[str, Path] = {}
    img_paths: List[Path] = []
    both_paths: List[Path] = []

    skipped_dup = 0
    skipped_missing = 0

    for p in img_paths_all:
        fn = p.name

        if fn in seen:
            skipped_dup += 1
            print(f"[WARN] duplicate filename in images/{split}: {fn} | {seen[fn]} and {p} -> skip {p}")
            continue
        seen[fn] = p

        hm = both_gt_dir / fn
        if not hm.exists():
            skipped_missing += 1
            print(f"[WARN] missing BOTH GT in {run_tag} -> skip: {fn}")
            continue

        img_paths.append(p)
        both_paths.append(hm)

    if len(img_paths) == 0:
        raise RuntimeError(f"No usable pairs found for split={split}. Check filenames & BOTH GT dir.")

    print(f"[PAIR] split={split} | usable={len(img_paths)} | skipped_missing={skipped_missing} | skipped_dup={skipped_dup}")
    return img_paths, both_paths


# -----------------------
# Dataset
# -----------------------
class Heatmap2CHDataset(Dataset):
    """
    returns:
      image: (3,H,W) tensor
      mask : (2,H,W) tensor  [0]=disc [1]=macula
      meta : {'filename': str}
    """
    def __init__(self, image_paths: List[Path], both_paths: List[Path], transform=None):
        assert len(image_paths) == len(both_paths)
        self.image_paths = image_paths
        self.both_paths  = both_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"Failed to read image: {img_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        disc01, mac01 = read_both_rg01(self.both_paths[idx])
        mask_2ch = np.stack([disc01, mac01], axis=-1).astype(np.float32)  # (H,W,2)

        if self.transform:
            aug = self.transform(image=rgb, mask=mask_2ch)
            image = aug["image"]  # (3,H,W)
            mask  = aug["mask"]   # (H,W,2) tensor usually
        else:
            image = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
            mask  = torch.from_numpy(mask_2ch)

        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask)
        if mask.ndim != 3 or mask.shape[-1] != 2:
            raise ValueError(f"Unexpected mask shape: {mask.shape}")

        mask = mask.permute(2, 0, 1).float()  # (2,H,W)
        meta = {"filename": img_path.name}
        return image, mask, meta


def collate_fn(batch):
    images = torch.stack([b[0] for b in batch], dim=0)
    masks  = torch.stack([b[1] for b in batch], dim=0)
    metas  = [b[2] for b in batch]
    return images, masks, metas


# -----------------------
# Save predictions: run_dir/prediction/{filename}.png (BOTH in one image)
# -----------------------
def save_predictions_both_png(
    model: torch.nn.Module,
    loader: DataLoader,
    device: str,
    run_dir: Path,
):
    """
    Save ONE PNG per sample under:
      run_dir/prediction/{filename}.png

    PNG encoding (OpenCV BGR):
      R(channel 2) = disc prediction  (0~255)
      G(channel 1) = macula prediction(0~255)
      B(channel 0) = 0
    """
    pred_dir = run_dir / "prediction"
    pred_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    with torch.no_grad():
        for images, _, metas in tqdm(loader, desc=f"Saving BOTH predictions -> {pred_dir}"):
            images = images.to(device)
            logits = model(images)          # (B,2,H,W)
            probs  = torch.sigmoid(logits)  # (B,2,H,W)

            bs = images.size(0)
            for i in range(bs):
                fn = metas[i]["filename"]

                pr_disc = probs[i, 0].detach().cpu().numpy()  # [0,1]
                pr_mac  = probs[i, 1].detach().cpu().numpy()  # [0,1]

                g = (np.clip(pr_disc, 0, 1) * 255.0).astype(np.uint8)  # disc -> G
                r = (np.clip(pr_mac,  0, 1) * 255.0).astype(np.uint8)  # mac  -> R
                b = np.zeros_like(r, dtype=np.uint8)

                both_bgr = np.stack([b, g, r], axis=-1)  # (H,W,3) BGR
                cv2.imwrite(str(pred_dir / fn), both_bgr)

    print(f"[SAVE] BOTH pred PNG saved: {pred_dir}")


# -----------------------
# Debug subplot (only when --debug)
# -----------------------
def save_debug_subplot_vis(
    model: torch.nn.Module,
    loader: DataLoader,
    device: str,
    debug_dir: Path,
    overlay_alpha: float,
):
    debug_dir.mkdir(parents=True, exist_ok=True)
    model.eval()

    count = 0
    with torch.no_grad():
        for images, masks, metas in tqdm(loader, desc="Debug vis (subplot)"):
            images = images.to(device)
            masks  = masks.to(device)

            logits = model(images)
            probs  = torch.sigmoid(logits)

            bs = images.size(0)
            for i in range(bs):
                fn = metas[i]["filename"]

                orig_rgb01 = denormalize(images[i])
                gt_disc = masks[i, 0].detach().cpu().numpy()
                gt_mac  = masks[i, 1].detach().cpu().numpy()
                pr_disc = probs[i, 0].detach().cpu().numpy()
                pr_mac  = probs[i, 1].detach().cpu().numpy()

                gt_overlay = overlay_two_heatmaps_rgb(orig_rgb01, gt_disc, gt_mac, alpha=overlay_alpha)
                pr_overlay = overlay_two_heatmaps_rgb(orig_rgb01, pr_disc, pr_mac, alpha=overlay_alpha)

                fig, axes = plt.subplots(2, 4, figsize=(16, 8))
                axes[0, 0].imshow(orig_rgb01); axes[0, 0].set_title(f"Original\n{fn}")
                axes[0, 1].imshow(gt_disc, cmap="jet"); axes[0, 1].set_title("GT Disc")
                axes[0, 2].imshow(gt_mac, cmap="hot");  axes[0, 2].set_title("GT Macula")
                axes[0, 3].imshow(gt_overlay);          axes[0, 3].set_title("GT Overlay")

                axes[1, 0].imshow(pr_overlay);          axes[1, 0].set_title("Pred Overlay")
                axes[1, 1].imshow(pr_disc, cmap="jet"); axes[1, 1].set_title("Pred Disc")
                axes[1, 2].imshow(pr_mac, cmap="hot");  axes[1, 2].set_title("Pred Macula")
                axes[1, 3].imshow(np.maximum(pr_disc, pr_mac), cmap="gray"); axes[1, 3].set_title("Pred max")

                for ax in axes.ravel():
                    ax.axis("off")

                out = debug_dir / f"result_{count:05d}.png"
                plt.savefig(out, bbox_inches="tight", pad_inches=0.1)
                plt.close(fig)
                count += 1

    print(f"[DEBUG] subplot vis saved: {debug_dir} (count={count})")


# -----------------------
# Main
# -----------------------
def main():
    import argparse

    p = argparse.ArgumentParser()

    # data
    p.add_argument("--det_root", type=str, required=True,
                   help="Root containing images/{train,valid,test} and heatmap/heatmap/{test_case}_{split}/")
    p.add_argument("--recursive_images", action="store_true",
                   help="If images/{split} has subfolders, search recursively.")

    # run
    p.add_argument("--test_case", type=str, required=True,
                   help="Used in BOTH GT folder name: heatmap/heatmap/{test_case}_{split}/")
    p.add_argument("--save_root", type=str, default="/mnt/richul_FM/UWF_seg_det/datasets/Det/train_output",
                   help="Where to create {test_case} folder")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--thr", type=float, default=0.5)
    p.add_argument("--overlay_alpha", type=float, default=0.45)

    # debug
    p.add_argument("--debug", action="store_true",
                   help="If set, save plt.subplot comparisons into run_dir/debug_vis/")

    args = p.parse_args()

    det_root = Path(args.det_root)
    save_root = Path(args.save_root)

    run_dir = save_root / args.test_case
    run_dir.mkdir(parents=True, exist_ok=True)

    device = CONFIG["DEVICE"]
    print(f"[DEVICE]   {device}")
    print(f"[DET_ROOT] {det_root}")
    print(f"[RUN_DIR]  {run_dir}")

    ckpt_path = run_dir / f"{args.test_case}_best_model.pth"
    print(f"[CKPT]     {ckpt_path}")

    # transforms
    train_transform = A.Compose([
        A.Resize(height=CONFIG["IMG_SIZE"][0], width=CONFIG["IMG_SIZE"][1]),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=30, p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    val_transform = A.Compose([
        A.Resize(height=CONFIG["IMG_SIZE"][0], width=CONFIG["IMG_SIZE"][1]),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    # pairs (BOTH GT PATH)
    train_img, train_both = build_pairs_for_split_both(
        det_root=det_root, split="train",
        recursive_images=args.recursive_images, test_case=args.test_case
    )

    valid_img, valid_both = build_pairs_for_split_both(
        det_root=det_root, split="valid",
        recursive_images=args.recursive_images, test_case=args.test_case
    )

    train_ds = Heatmap2CHDataset(train_img, train_both, transform=train_transform)
    valid_ds = Heatmap2CHDataset(valid_img, valid_both, transform=val_transform)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=CONFIG["NUM_WORKERS"], collate_fn=collate_fn
    )
    valid_loader = DataLoader(
        valid_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=CONFIG["NUM_WORKERS"], collate_fn=collate_fn
    )

    # model (2ch)
    model = smp.Unet(
        encoder_name=CONFIG["ENCODER"],
        encoder_weights=CONFIG["WEIGHTS"],
        in_channels=3,
        classes=2,
        activation=None,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_loss = float("inf")

    # training
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0

        for images, masks, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]"):
            images = images.to(device)
            masks  = masks.to(device)

            optimizer.zero_grad()
            logits = model(images)
            probs  = torch.sigmoid(logits)

            loss = criterion(probs, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / max(1, len(train_loader))

        # validation
        model.eval()
        val_loss = 0.0

        disc_iou_sum = disc_dice_sum = 0.0
        mac_iou_sum  = mac_dice_sum  = 0.0
        micro_iou_sum = micro_dice_sum = 0.0

        with torch.no_grad():
            for images, masks, _ in tqdm(valid_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Valid]"):
                images = images.to(device)
                masks  = masks.to(device)

                logits = model(images)
                probs  = torch.sigmoid(logits)

                loss = criterion(probs, masks)
                val_loss += loss.item()

                bs = images.size(0)
                for i in range(bs):
                    di, dd = calculate_metrics_binary(probs[i, 0], masks[i, 0], threshold=args.thr)
                    mi, md = calculate_metrics_binary(probs[i, 1], masks[i, 1], threshold=args.thr)
                    mic_i, mic_d = calculate_micro_metrics(probs[i], masks[i], threshold=args.thr)

                    disc_iou_sum += di
                    disc_dice_sum += dd
                    mac_iou_sum  += mi
                    mac_dice_sum += md
                    micro_iou_sum += mic_i
                    micro_dice_sum += mic_d

        avg_val_loss = val_loss / max(1, len(valid_loader))
        n = max(1, len(valid_ds))

        disc_iou  = disc_iou_sum / n
        disc_dice = disc_dice_sum / n
        mac_iou   = mac_iou_sum / n
        mac_dice  = mac_dice_sum / n
        macro_iou  = 0.5 * (disc_iou + mac_iou)
        macro_dice = 0.5 * (disc_dice + mac_dice)
        micro_iou  = micro_iou_sum / n
        micro_dice = micro_dice_sum / n

        print(f"\nEpoch [{epoch+1}/{args.epochs}]")
        print(f"  Train Loss:   {avg_train_loss:.6f}")
        print(f"  Valid Loss:   {avg_val_loss:.6f}")
        print(f"  [DISC ] IoU: {disc_iou:.4f} | Dice: {disc_dice:.4f}")
        print(f"  [MAC  ] IoU: {mac_iou:.4f} | Dice: {mac_dice:.4f}")
        print(f"  [MACRO] IoU: {macro_iou:.4f} | Dice: {macro_dice:.4f}")
        print(f"  [MICRO] IoU: {micro_iou:.4f} | Dice: {micro_dice:.4f}\n")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), str(ckpt_path))
            print(f"  => Best saved: {ckpt_path} (val_loss={best_val_loss:.6f})")

    print("[TRAIN] Finished.")

    # load best
    if ckpt_path.exists():
        model.load_state_dict(torch.load(str(ckpt_path), map_location=device))
        model.to(device)
        model.eval()
        print(f"[LOAD] Best checkpoint loaded: {ckpt_path}")
    else:
        print("[WARN] Best checkpoint not found; using last weights.")

    # run_dir/prediction/ 에 저장
    save_predictions_both_png(model=model, loader=valid_loader, device=device, run_dir=run_dir)

    # debug subplot only when --debug
    if args.debug:
        debug_dir = run_dir / "debug_vis"
        save_debug_subplot_vis(
            model=model, loader=valid_loader, device=device,
            debug_dir=debug_dir, overlay_alpha=args.overlay_alpha
        )

    print("[DONE]")


if __name__ == "__main__":
    main()
