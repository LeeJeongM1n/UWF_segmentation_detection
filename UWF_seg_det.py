
# output file name format: OCT_0001_px12.34_det1_conf0.876.png

import os
import sys
# UWF_Quality 폴더에서 SmpModel/denorm_img 가져오기
sys.path.append("/mnt/richul_FM/UWF_Quality")
from train import SmpModel, denorm_img

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

# -------------------------------
# train.py에 있는 SmpModel/denorm_img 가져오기
# -------------------------------
from train import SmpModel, denorm_img  # train.py와 같은 폴더에 joint_inference.py가 있다고 가정


# ============================================================
# Segmentation 쪽 transform (Resize + ToTensor + Normalize)
# ============================================================
def build_seg_transform(input_size):
    """
    input_size: int 또는 (H, W)
    """
    if isinstance(input_size, int):
        resize_size = (input_size, input_size)
    elif isinstance(input_size, (tuple, list)) and len(input_size) == 2:
        resize_size = tuple(input_size)
    else:
        raise ValueError("--seg_input_size should be int or 2-length list/tuple")

    tf = transforms.Compose([
        transforms.Resize(resize_size, interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
    ])
    return tf


# ============================================================
# YOLOv5 로딩 및 inference 유틸
# ============================================================
def load_yolo_model(weights_path: str, device_str: str = "0", imgsz: int = 640):
    """
    Load YOLOv5 DetectMultiBackend 
    weights_path : pretrained best.pt path
    device_str   : '0' / 'cpu' str type
    imgsz        : YOLO (imgsz * imgsz)
    """
    yolo_dir = Path("YOLO/yolov5")
    if not yolo_dir.exists():
        raise FileNotFoundError

    sys.path.append(str(yolo_dir))

    from models.common import DetectMultiBackend
    from utils.torch_utils import select_device

    device = select_device(device_str)
    model = DetectMultiBackend(weights_path, device=device, dnn=False, fp16=False)
    stride = model.stride
    names = model.names
    return model, device, stride, names


def run_yolo_on_image(model, device, stride, img0, conf_thres=0.25, iou_thres=0.45, imgsz=640):
    """
    img0: BGR format
    반환: found (0 or 1), best_conf (float), best_box (x1,y1,x2,y2) or None
    """
    from utils.augmentations import letterbox
    from utils.general import non_max_suppression, scale_boxes

    # letterbox resize
    img = letterbox(img0, imgsz, stride=stride, auto=True)[0]
    # BGR -> RGB, HWC -> CHW
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)

    img_tensor = torch.from_numpy(img).to(device)
    img_tensor = img_tensor.float() / 255.0
    if img_tensor.ndim == 3:
        img_tensor = img_tensor.unsqueeze(0)  # [1,3,H,W]

    with torch.no_grad():
        pred = model(img_tensor, augment=False, visualize=False)

    # NMS
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes=None, agnostic=False, max_det=100)
    det = pred[0]

    if det is None or len(det) == 0:
        return 0, 0.0, None

    # 가장 confidence 높은 한 개 선택
    best_idx = det[:, 4].argmax()
    best_det = det[best_idx]

    # 좌표 rescale
    best_det = best_det.unsqueeze(0)
    best_det[:, :4] = scale_boxes(img_tensor.shape[2:], best_det[:, :4], img0.shape[:2]).round()

    x1, y1, x2, y2, conf, cls = best_det[0].tolist()
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    return 1, float(conf), (x1, y1, x2, y2)


# ============================================================
# Segmentation inference
# ============================================================
def load_seg_model(ckpt_path: str, device: torch.device):
    """
    Load pretrained SmpModel checkpoint (train.py)
    """
    model = SmpModel.load_from_checkpoint(ckpt_path, map_location=device)
    model.to(device)
    model.eval()
    return model


def run_seg_on_image(model, device, pil_img, seg_tf, thr=0.5):
    """
    pil_img : PIL.Image (RGB)
    seg_tf  : build_seg_transform 
    thr     : binarization threshold
    return  : mask_bool (H_orig, W_orig) np.bool_ array
    """
    w, h = pil_img.size
    img_input = seg_tf(pil_img)          # [3,H,W]
    img_input = img_input.unsqueeze(0).to(device)  # [1,3,H,W]

    with torch.no_grad():
        prob = model(img_input)          # [1,1,h',w'] in [0,1]
    if isinstance(prob, (tuple, list)):
        prob = prob[0]
    prob = prob.squeeze().detach().cpu().numpy()   # [h', w']

    # threshold
    pred_bin = (prob > thr).astype(np.uint8)

    # 원본 해상도로 리사이즈
    mask_resized = cv2.resize(
        pred_bin,
        (w, h),  # (width, height)
        interpolation=cv2.INTER_NEAREST
    )
    mask_bool = mask_resized.astype(bool)
    return mask_bool


# ============================================================
# Overlay 생성
# ============================================================
def make_overlay(orig_bgr, mask_bool, yolo_info, seg_alpha=0.3):
    """
    orig_bgr : BGR (H,W,3)
    mask_bool: segmentation mask (H,W) bool
    yolo_info: (found, conf, box) from run_yolo_on_image
    seg_alpha: segmentation overlay alpha (0.3)
    """
    overlay = orig_bgr.copy()

    # segmentation overlay (빨간색)
    color_seg = np.array([0, 0, 255], dtype=np.uint8)  # BGR
    seg_idx = mask_bool
    if seg_idx.any():
        # seg 픽셀에만 색깔 섞기
        base = overlay[seg_idx]
        overlay[seg_idx] = (base * (1.0 - seg_alpha) + color_seg * seg_alpha).astype(np.uint8)

    # YOLO bbox + label
    found, conf, box = yolo_info
    if found and box is not None:
        x1, y1, x2, y2 = box
        # 빨간색 박스
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), 2)

        label = f"disc {conf:.2f}"
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        y_text = max(y1 - 5, th + 5)
        cv2.rectangle(overlay, (x1, y_text - th - 5), (x1 + tw + 5, y_text + baseline - 5), (0, 0, 255), -1)
        cv2.putText(
            overlay,
            label,
            (x1 + 2, y_text - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            lineType=cv2.LINE_AA,
        )

    return overlay


# ============================================================
# 메인 루프
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--image_dir", type=str, required=True,
                        help="입력 이미지 폴더 (png/jpg/jpeg 확장자 자동 검색)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="결과를 저장할 폴더")

    # YOLO
    parser.add_argument("--yolo_weights", type=str, required=True,
                        help="YOLOv5 학습 weight (예: runs/train/exp/weights/best.pt)")
    parser.add_argument("--yolo_device", type=str, default="0",
                        help="YOLO device ('0', '1', 'cpu' 등)")
    parser.add_argument("--yolo_img_size", type=int, default=640,
                        help="YOLO 입력 해상도 (정사각)")

    # Segmentation
    parser.add_argument("--seg_ckpt", type=str, required=True,
                        help="SmpModel checkpoint (.ckpt)")
    parser.add_argument("--seg_device", type=str, default="0",
                        help="'cuda' or 'cpu'")
    parser.add_argument("--seg_input_size", nargs="+", type=int, default=[320, 320],
                        help="Segmentation 입력 크기 [H W] 또는 [S]")

    parser.add_argument("--seg_thr", type=float, default=0.5,
                        help="Segmentation binarization threshold")
    parser.add_argument("--seg_alpha", type=float, default=0.3,
                        help="Segmentation overlay alpha (0~1)")

    return parser.parse_args()


def main():
    args = parse_args()

    image_dir = Path(args.image_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.seg_device
    seg_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    seg_model = load_seg_model(args.seg_ckpt, seg_device)

    if len(args.seg_input_size) == 1:
        seg_input_size = int(args.seg_input_size[0])
    else:
        seg_input_size = (args.seg_input_size[0], args.seg_input_size[1])

    seg_tf = build_seg_transform(seg_input_size)

    # ---------------- YOLO 모델 ----------------
    yolo_model, yolo_device, yolo_stride, yolo_names = load_yolo_model(
        weights_path=args.yolo_weights,
        device_str=args.yolo_device,
        imgsz=args.yolo_img_size,
    )

    # ---------------- 이미지 리스트 ----------------
    exts = [".png", ".jpg"]
    image_paths = sorted(
        [p for p in image_dir.rglob("*") if p.suffix.lower() in exts]
    )
    if not image_paths:
        print(f"[ERROR] no images contained in {image_dir}.")
        return

    print(f"[INFO] # of images : {len(image_paths)}")

    for img_path in image_paths:
        print(f"[PROC] {img_path}")

        # --- 원본 이미지 (BGR, RGB 둘 다 사용) ---
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            print(f"[WARN] missing: {img_path}")
            continue

        h, w = img_bgr.shape[:2]
        pil_img = Image.open(str(img_path)).convert("RGB")

        # --- Segmentation ---
        mask_bool = run_seg_on_image(
            seg_model,
            seg_device,
            pil_img,
            seg_tf,
            thr=args.seg_thr
        )
        roi_pixels = int(mask_bool.sum())
        total_pixels = int(h * w) if h > 0 and w > 0 else 1
        pixel_percent = 100.0 * roi_pixels / total_pixels

        # --- YOLO detection ---
        found, conf, box = run_yolo_on_image(
            yolo_model,
            yolo_device,
            yolo_stride,
            img_bgr,
            conf_thres=0.25,
            iou_thres=0.45,
            imgsz=args.yolo_img_size
        )

        # --- Overlay 생성 ---
        overlay = make_overlay(
            orig_bgr=img_bgr,
            mask_bool=mask_bool,
            yolo_info=(found, conf, box),
            seg_alpha=args.seg_alpha
        )

        # --- 파일 이름 구성 및 저장 ---
        stem = img_path.stem
        save_name = f"{stem}_px{pixel_percent:.2f}_det{found}_conf{conf:.3f}.png"
        save_path = output_dir / save_name

        cv2.imwrite(str(save_path), overlay)
        print(f"  -> saved: {save_path}")

    print("[DONE]")


if __name__ == "__main__":
    main()
