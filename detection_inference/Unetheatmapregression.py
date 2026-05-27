import os
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import cv2
import segmentation_models_pytorch as smp
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import albumentations as A
from albumentations.pytorch import ToTensorV2


# ==========================================
# 1. 설정 (Configuration) - 기본값
# ==========================================
CONFIG = {
    "IMG_SIZE": (512, 512),        # 입력 이미지 크기 (Resize)
    "BATCH_SIZE": 8,
    "LEARNING_RATE": 1e-4,
    "EPOCHS": 20,
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "ENCODER": "efficientnet-b0",  # 원본 코드 기본값 유지
    "WEIGHTS": "imagenet",
    "NUM_WORKERS": 4,
}

# ImageNet 정규화 복구용 (시각화를 위해 필수)
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ==========================================
# 2. 데이터셋 클래스 (Dataset)
# ==========================================
class HeatmapRegressionDataset(Dataset):
    def __init__(self, image_paths, heatmap_paths, transform=None):
        self.image_paths = image_paths
        self.heatmap_paths = heatmap_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 1) Image (RGB)
        img_path = self.image_paths[idx]
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Failed to read image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 2) Heatmap (Grayscale)
        mask_path = self.heatmap_paths[idx]
        heatmap = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if heatmap is None:
            raise FileNotFoundError(f"Failed to read heatmap: {mask_path}")

        # normalize 0~255 -> 0~1 float
        heatmap = heatmap.astype(np.float32) / 255.0

        # 3) Transform
        if self.transform:
            augmented = self.transform(image=image, mask=heatmap)
            image = augmented["image"]
            heatmap = augmented["mask"]

        # 4) shape: mask (H,W) -> (1,H,W)
        if heatmap.ndim == 2:
            heatmap = heatmap.unsqueeze(0)

        return image, heatmap


# ==========================================
# 3. 파일 페어링
# ==========================================
def get_file_pairs(img_dir, mask_dir):
    """
    이미지 폴더의 파일을 기준으로 동일한 이름의 히트맵 경로를 생성하여 쌍을 맞춥니다.
    (원본 코드의 로직 유지: 확장자 제한 + 파일명 동일 가정)
    """
    img_paths = []
    mask_paths = []

    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp"}

    if not os.path.isdir(img_dir):
        raise FileNotFoundError(f"Image directory not found: {img_dir}")
    if not os.path.isdir(mask_dir):
        raise FileNotFoundError(f"Heatmap directory not found: {mask_dir}")

    files = os.listdir(img_dir)
    files.sort()

    for filename in files:
        ext = os.path.splitext(filename)[1].lower()
        if ext not in valid_extensions:
            continue

        img_path = os.path.join(img_dir, filename)
        mask_path = os.path.join(mask_dir, filename)

        if os.path.exists(mask_path):
            img_paths.append(img_path)
            mask_paths.append(mask_path)
        else:
            print(f"[Warning] 히트맵을 찾을 수 없음: {filename} (제외됨)")

    return img_paths, mask_paths


# ==========================================
# 4. Metrics (IoU & Dice)
# ==========================================
def calculate_metrics(pred, target, threshold=0.5):
    """
    pred: sigmoid 통과된 0~1 텐서 (shape: [1,H,W] or [H,W])
    target: 0~1 텐서
    """
    pred_mask = (pred > threshold).float()
    target_mask = (target > threshold).float()

    intersection = (pred_mask * target_mask).sum()
    union = pred_mask.sum() + target_mask.sum() - intersection

    if union == 0:
        iou = 0.0
    else:
        iou = intersection / union

    dice_den = pred_mask.sum() + target_mask.sum()
    if dice_den == 0:
        dice = 0.0
    else:
        dice = (2 * intersection) / dice_den

    return float(iou.item()), float(dice.item())


# ==========================================
# 5. Visualization helpers
# ==========================================
def denormalize(tensor_chw):
    """Normalized된 텐서를 원본 RGB numpy 이미지(0~1 float)로 복구"""
    img = tensor_chw.permute(1, 2, 0).detach().cpu().numpy()
    img = img * STD + MEAN
    return np.clip(img, 0, 1)

def apply_heatmap_overlay(image, heatmap):
    """이미지 위에 히트맵(Jet Color)을 오버레이"""
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    # image float(0~1) -> uint8
    if image.dtype != np.uint8:
        image_u8 = (image * 255).astype(np.uint8)
    else:
        image_u8 = image

    return cv2.addWeighted(image_u8, 0.6, heatmap_color, 0.4, 0)

def make_comparison_mask(gt_mask, pred_mask):
    """
    GT(초록) vs Pred(빨강)
    - 노랑: 겹침
    - 초록: GT만
    - 빨강: Pred만
    """
    h, w = gt_mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.float32)
    color_mask[:, :, 1] = gt_mask
    color_mask[:, :, 0] = pred_mask
    return color_mask


def save_visualization_results(model, loader, device, save_dir, threshold=0.5):
    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    count = 0
    print(f"Results will be saved to: {save_dir}")

    with torch.no_grad():
        for images, masks in tqdm(loader, desc="Saving Results"):
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            preds = torch.sigmoid(outputs)

            batch_size = images.size(0)
            for i in range(batch_size):
                orig_img = denormalize(images[i])  # float 0~1 RGB
                gt_heatmap = masks[i].squeeze().cpu().numpy()
                gt_binary = (gt_heatmap > threshold).astype(np.float32)

                pred_heatmap = preds[i].squeeze().cpu().numpy()
                pred_binary = (pred_heatmap > threshold).astype(np.float32)

                gt_overlay = apply_heatmap_overlay(orig_img, gt_heatmap)
                pred_overlay = apply_heatmap_overlay(orig_img, pred_heatmap)

                comp_mask = make_comparison_mask(gt_binary, pred_binary)

                fig, axes = plt.subplots(1, 8, figsize=(24, 4))

                axes[0].imshow(orig_img);                axes[0].set_title("1. Original")
                axes[1].imshow(gt_heatmap, cmap="jet");  axes[1].set_title("2. GT Heatmap")
                axes[2].imshow(gt_binary, cmap="gray");  axes[2].set_title("3. GT Binary")
                axes[3].imshow(pred_heatmap, cmap="jet");axes[3].set_title("4. Pred Heatmap")
                axes[4].imshow(pred_binary, cmap="gray");axes[4].set_title("5. Pred Binary")
                axes[5].imshow(gt_overlay);              axes[5].set_title("6. GT Overlay")
                axes[6].imshow(pred_overlay);            axes[6].set_title("7. Pred Overlay")
                axes[7].imshow(comp_mask);               axes[7].set_title("8. G=GT, R=Pred, Y=Both")

                for ax in axes:
                    ax.axis("off")

                save_path = os.path.join(save_dir, f"result_{count:03d}.png")
                plt.savefig(save_path, bbox_inches="tight", pad_inches=0.1)
                plt.close(fig)
                count += 1

    print(f"모든 이미지 저장이 완료되었습니다. '{save_dir}' 폴더를 확인하세요.")


# ==========================================
# 6. main()
# ==========================================
def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True, help="DATA_ROOT (dataset root path)")
    parser.add_argument("--sigma", type=str, required=True, help="sigma token for folder name (e.g., 10, 12.5)")
    parser.add_argument("--target", type=str, required=True, help="disc or macula")
    parser.add_argument("--test_case", type=str, required=True, help="prefix for best checkpoint filename")
    args = parser.parse_args()

    data_root = args.data_root
    test_case = args.test_case
    device = CONFIG["DEVICE"]

    # transforms (원본과 동일)
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

    # dataset paths (요구사항 반영: sigma{sigma}_train / sigma{sigma}_test)
    train_img_dir = os.path.join(data_root, "images", "train")
    train_mask_dir = os.path.join(
        data_root, "heatmap", args.target,  "heatmap", f"sigma{args.sigma}_train"
    )

    val_img_dir = os.path.join(data_root, "images", "test")
    val_mask_dir = os.path.join(
        data_root, "heatmap", args.target,  "heatmap", f"sigma{args.sigma}_test"
    )

    train_img_paths, train_mask_paths = get_file_pairs(train_img_dir, train_mask_dir)
    val_img_paths, val_mask_paths = get_file_pairs(val_img_dir, val_mask_dir)

    print(f"학습 데이터: {len(train_img_paths)}쌍 로드됨")
    print(f"검증 데이터: {len(val_img_paths)}쌍 로드됨")

    train_dataset = HeatmapRegressionDataset(train_img_paths, train_mask_paths, transform=train_transform)
    val_dataset = HeatmapRegressionDataset(val_img_paths, val_mask_paths, transform=val_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG["BATCH_SIZE"],
        shuffle=True,
        num_workers=CONFIG["NUM_WORKERS"],
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG["BATCH_SIZE"],
        shuffle=False,
        num_workers=CONFIG["NUM_WORKERS"],
    )

    # model/optim (원본 유지)
    model = smp.Unet(
        encoder_name=CONFIG["ENCODER"],
        encoder_weights=CONFIG["WEIGHTS"],
        in_channels=3,
        classes=1,
        activation=None,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["LEARNING_RATE"])

    print(f"Start Training on {device}...")

    best_val_loss = float("inf")
    save_dir = "/mnt/richul_FM/YOLO/evaluation_results"
    os.makedirs(save_dir, exist_ok=True)

    best_ckpt_name = f"{test_case}_best_model.pth"
    best_ckpt_path = os.path.join(save_dir, best_ckpt_name)

    for epoch in range(CONFIG["EPOCHS"]):
        # --- Train ---
        model.train()
        train_loss = 0.0

        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['EPOCHS']} [Train]"):
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(torch.sigmoid(outputs), masks)

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / max(1, len(train_loader))

        # --- Validation & Metrics ---
        model.eval()
        val_loss = 0.0
        val_iou = 0.0
        val_dice = 0.0

        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc=f"Epoch {epoch+1}/{CONFIG['EPOCHS']} [Val]"):
                images = images.to(device)
                masks = masks.to(device)

                outputs = model(images)
                probs = torch.sigmoid(outputs)

                loss = criterion(probs, masks)
                val_loss += loss.item()

                batch_iou = 0.0
                batch_dice = 0.0
                batch_size = images.size(0)

                for i in range(batch_size):
                    iou, dice = calculate_metrics(probs[i], masks[i], threshold=0.5)
                    batch_iou += iou
                    batch_dice += dice

                val_iou += (batch_iou / batch_size)
                val_dice += (batch_dice / batch_size)

        avg_val_loss = val_loss / max(1, len(val_loader))
        avg_val_iou = val_iou / max(1, len(val_loader))
        avg_val_dice = val_dice / max(1, len(val_loader))

        print(f"Epoch [{epoch+1}/{CONFIG['EPOCHS']}]")
        print(f"  Train Loss: {avg_train_loss:.6f}")
        print(f"  Val Loss  : {avg_val_loss:.6f} | IoU: {avg_val_iou:.4f} | Dice: {avg_val_dice:.4f}")

        # best 저장
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_ckpt_path)
            print(f"  => Best Model Saved! -> {best_ckpt_path} (Loss: {best_val_loss:.6f})")

    print("Training Finished.")

    # 시각화 저장 (val_loader 기준)
    save_dir = os.path.join("/mnt/richul_FM/YOLO/evaluation_results", test_case)
    save_visualization_results(model, val_loader, device, save_dir=save_dir, threshold=0.5)


if __name__ == "__main__":
    main()
