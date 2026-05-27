# -*- coding: utf-8 -*-
# UWF_seg_det_2class_metrics.py
# - inference + (optional) evaluation (seg Dice/IoU, det TP/FP/FN + mAP50)
# - INFER CSV: 기존과 동일한 컬럼/순서 유지
# - EVAL CSV: 별도 저장

import os
import sys
import argparse
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image

import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

import pandas as pd

# -------------------------------
# UWF_Quality 폴더에서 SmpModel/denorm_img 가져오기
# -------------------------------
sys.path.append("/mnt/richul_FM/UWF_Quality")
from train import SmpModel, denorm_img  # noqa


# ============================================================
# Segmentation transform
# ============================================================
def build_seg_transform(input_size):
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
# YOLOv5 loading & inference
# ============================================================
def load_yolo_model(weights_path: str, device_str: str = "0", imgsz: int = 640):
    yolo_dir = Path("YOLO/yolov5")
    if not yolo_dir.exists():
        raise FileNotFoundError(f"YOLOv5 directory not found: {yolo_dir}")

    sys.path.append(str(yolo_dir))
    from models.common import DetectMultiBackend
    from utils.torch_utils import select_device

    device = select_device(device_str)
    model = DetectMultiBackend(weights_path, device=device, dnn=False, fp16=False)
    stride = model.stride
    names = model.names
    return model, device, stride, names


def run_yolo_on_image(model, device, stride, img0, conf_thres=0.25, iou_thres=0.45, imgsz=640):
    from utils.augmentations import letterbox
    from utils.general import non_max_suppression, scale_boxes

    img = letterbox(img0, imgsz, stride=stride, auto=True)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)

    img_tensor = torch.from_numpy(img).to(device)
    img_tensor = img_tensor.float() / 255.0
    if img_tensor.ndim == 3:
        img_tensor = img_tensor.unsqueeze(0)

    with torch.no_grad():
        pred = model(img_tensor, augment=False, visualize=False)

    pred = non_max_suppression(pred, conf_thres, iou_thres, classes=None, agnostic=False, max_det=100)
    det = pred[0]

    results = {
        0: (0, 0.0, None),  # disc
        1: (0, 0.0, None),  # macula
    }
    if det is None or len(det) == 0:
        return results

    for class_id in [0, 1]:
        det_cls = det[det[:, 5] == class_id]
        if det_cls is None or len(det_cls) == 0:
            continue

        best_idx = det_cls[:, 4].argmax()
        best_det = det_cls[best_idx].unsqueeze(0)

        best_det[:, :4] = scale_boxes(img_tensor.shape[2:], best_det[:, :4], img0.shape[:2]).round()

        x1, y1, x2, y2, conf, cls = best_det[0].tolist()
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        results[class_id] = (1, float(conf), (x1, y1, x2, y2))

    return results


def run_yolo_on_image_all(model, device, stride, img0, conf_thres=0.001, iou_thres=0.45, imgsz=640):
    """mAP 계산용: NMS 통과한 전체 박스 반환 [(cls, conf, (x1,y1,x2,y2)), ...]"""
    from utils.augmentations import letterbox
    from utils.general import non_max_suppression, scale_boxes

    img = letterbox(img0, imgsz, stride=stride, auto=True)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)

    img_tensor = torch.from_numpy(img).to(device)
    img_tensor = img_tensor.float() / 255.0
    if img_tensor.ndim == 3:
        img_tensor = img_tensor.unsqueeze(0)

    with torch.no_grad():
        pred = model(img_tensor, augment=False, visualize=False)

    pred = non_max_suppression(pred, conf_thres, iou_thres, classes=None, agnostic=False, max_det=300)
    det = pred[0]
    if det is None or len(det) == 0:
        return []

    det[:, :4] = scale_boxes(img_tensor.shape[2:], det[:, :4], img0.shape[:2]).round()
    out = []
    for *xyxy, conf, cls in det.tolist():
        x1, y1, x2, y2 = map(int, xyxy)
        out.append((int(cls), float(conf), (x1, y1, x2, y2)))
    return out


# ============================================================
# Segmentation inference
# ============================================================
def load_seg_model(ckpt_path: str, device: torch.device):
    model = SmpModel.load_from_checkpoint(ckpt_path, map_location=device)
    model.to(device)
    model.eval()
    return model


def run_seg_on_image(model, device, pil_img, seg_tf, thr=0.5):
    w, h = pil_img.size
    img_input = seg_tf(pil_img)
    img_input = img_input.unsqueeze(0).to(device)

    with torch.no_grad():
        prob = model(img_input)
    if isinstance(prob, (tuple, list)):
        prob = prob[0]
    prob = prob.squeeze().detach().cpu().numpy()

    pred_bin = (prob > thr).astype(np.uint8)

    mask_resized = cv2.resize(pred_bin, (w, h), interpolation=cv2.INTER_NEAREST)
    return mask_resized.astype(bool)


# ============================================================
# Overlay
# ============================================================
# >>> CHANGED: macula_gt_boxes 인자 추가
def make_overlay(orig_bgr, mask_bool, yolo_info, seg_alpha=0.3, macula_gt_boxes=None):
    overlay = orig_bgr.copy()

    color_seg = np.array([0, 0, 255], dtype=np.uint8)  # BGR
    seg_idx = mask_bool
    if seg_idx.any():
        base = overlay[seg_idx]
        overlay[seg_idx] = (base * (1.0 - seg_alpha) + color_seg * seg_alpha).astype(np.uint8)

    class_names = {0: "disc", 1: "macula"}
    class_colors = {0: (0, 0, 255), 1: (0, 255, 0)}  # pred: disc=red, macula=green

    # pred boxes
    for cls_id, (found, conf, box) in yolo_info.items():
        if not found or box is None:
            continue
        x1, y1, x2, y2 = box
        color = class_colors.get(cls_id, (0, 0, 255))
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 5)

        label = f"{class_names.get(cls_id, str(cls_id))} {conf:.2f}"
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        y_text = max(y1 - 5, th + 5)
        cv2.rectangle(overlay, (x1, y_text - th - 5), (x1 + tw + 5, y_text + baseline - 5), color, -1)
        cv2.putText(overlay, label, (x1 + 2, y_text - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, lineType=cv2.LINE_AA)

    # >>> ADDED: macula GT boxes (class=1) in different color
    if macula_gt_boxes is not None and len(macula_gt_boxes) > 0:
        gt_color = (0, 0, 255)  # BGR: red
        for (x1, y1, x2, y2) in macula_gt_boxes:
            cv2.rectangle(overlay, (x1, y1), (x2, y2), gt_color, 4)

            label = "macula GT"
            (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            y_text = max(y1 - 5, th + 5)
            cv2.rectangle(overlay, (x1, y_text - th - 5), (x1 + tw + 5, y_text + baseline - 5), gt_color, -1)
            cv2.putText(overlay, label, (x1 + 2, y_text - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, lineType=cv2.LINE_AA)

    return overlay


# ============================================================
# >>> ADDED/USED: metadata loading (원래 방식 복원)
# ============================================================
def load_mshf_meta(xlsx_path: str) -> pd.DataFrame:
    df = pd.read_excel(xlsx_path, header=[0, 1])
    fname = df.iloc[:, 0].astype(str)

    ill_cols = [col for col in df.columns if 'annotator' in str(col[0]) and str(col[1]).lower() == 'illumination']
    cla_cols = [col for col in df.columns if 'annotator' in str(col[0]) and str(col[1]).lower() == 'clarity']
    con_cols = [col for col in df.columns if 'annotator' in str(col[0]) and str(col[1]).lower() == 'contrast']
    oq_cols  = [col for col in df.columns if 'annotator' in str(col[0]) and str(col[1]).lower() == 'overall']

    def majority_vote(cols):
        votes = df[cols].astype(float)
        return (votes.sum(axis=1) >= 2).astype(int)

    ILL = majority_vote(ill_cols) if len(ill_cols) else 0
    CLA = majority_vote(cla_cols) if len(cla_cols) else 0
    CON = majority_vote(con_cols) if len(con_cols) else 0
    OQ  = majority_vote(oq_cols)  if len(oq_cols)  else 0

    meta = pd.DataFrame({"fname": fname, "ILL": ILL, "CLA": CLA, "CON": CON, "OQ": OQ})
    meta["stem"] = meta["fname"].apply(lambda x: Path(x).stem)
    return meta


def load_ouwfd_meta(xlsx_path: str) -> pd.DataFrame:
    df = pd.read_excel(xlsx_path)
    df = df.rename(columns={
        "Image ID ": "fname",
        "Field of view": "FOV",
        "Contrast": "CON",
        "Illumination": "ILL",
        "Artifacts ": "ART",
        "Overall quality": "OQ",
    })
    df["fname"] = df["fname"].astype(str)
    df["stem"] = df["fname"].apply(lambda x: Path(x).stem)
    return df


# ============================================================
# Evaluation utilities
# ============================================================
def load_gt_mask_as_bool(mask_path: Path) -> np.ndarray:
    m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(f"GT mask not found/readable: {mask_path}")
    return (m > 0)


def dice_iou(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-7):
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    dice = (2.0 * inter + eps) / (pred.sum() + gt.sum() + eps)
    iou = (inter + eps) / (union + eps)
    return float(dice), float(iou)


def yolo_txt_to_boxes_xyxy(txt_path: Path, img_w: int, img_h: int):
    out: Dict[int, List[Tuple[int, int, int, int]]] = {}
    if not txt_path.exists():
        return out
    with open(txt_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            cls = int(float(parts[0]))
            cx, cy, bw, bh = map(float, parts[1:5])

            x1 = (cx - bw / 2.0) * img_w
            y1 = (cy - bh / 2.0) * img_h
            x2 = (cx + bw / 2.0) * img_w
            y2 = (cy + bh / 2.0) * img_h

            x1 = int(round(max(0, min(img_w - 1, x1))))
            y1 = int(round(max(0, min(img_h - 1, y1))))
            x2 = int(round(max(0, min(img_w - 1, x2))))
            y2 = int(round(max(0, min(img_h - 1, y2))))

            out.setdefault(cls, []).append((x1, y1, x2, y2))
    return out


def bbox_iou_xyxy(a, b, eps: float = 1e-7):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    a_area = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    b_area = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = a_area + b_area - inter
    return float((inter + eps) / (union + eps))


def eval_top1_det(pred_found: int, pred_box, gt_boxes: list, iou_thr: float):
    if len(gt_boxes) == 0:
        if pred_found:
            return 0, 1, 0, 0.0  # FP
        return 0, 0, 0, 0.0      # TN(미카운트)

    if (not pred_found) or (pred_box is None):
        return 0, 0, 1, 0.0      # FN

    best_iou = max(bbox_iou_xyxy(pred_box, g) for g in gt_boxes)
    if best_iou >= iou_thr:
        return 1, 0, 0, float(best_iou)
    return 0, 1, 0, float(best_iou)


def find_seg_gt_mask(img_path: Path, gt_seg_dir: Path) -> Optional[Path]:
    # naming: {stem}_Retina_O.png
    stem = img_path.stem
    cand = gt_seg_dir / f"{stem}_Retina_O.png"
    return cand if cand.exists() else None


# =====================
# mAP(AP) utilities
# =====================
def compute_ap(rec, prec):
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])
    return float(ap)


def compute_ap_per_class(all_preds, all_gts, class_id, iou_thr=0.5):
    npos = sum(len(all_gts.get(k, [])) for k in all_gts.keys())
    if npos == 0:
        return 0.0, 0, 0

    all_preds = sorted(all_preds, key=lambda x: x[1], reverse=True)

    tp = np.zeros(len(all_preds), dtype=np.float32)
    fp = np.zeros(len(all_preds), dtype=np.float32)

    matched = {k: np.zeros(len(all_gts.get(k, [])), dtype=bool) for k in all_gts.keys()}

    for i, (img_key, conf, pbox) in enumerate(all_preds):
        gt_list = all_gts.get(img_key, [])
        if len(gt_list) == 0:
            fp[i] = 1
            continue

        ious = [bbox_iou_xyxy(pbox, g) for g in gt_list]
        best_j = int(np.argmax(ious))
        best_iou = ious[best_j]

        if best_iou >= iou_thr and not matched[img_key][best_j]:
            tp[i] = 1
            matched[img_key][best_j] = True
        else:
            fp[i] = 1

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)

    rec = tp_cum / (npos + 1e-9)
    prec = tp_cum / (tp_cum + fp_cum + 1e-9)

    ap = compute_ap(rec, prec)
    return ap, int(tp_cum[-1]), int(fp_cum[-1])


# ============================================================
# args / main
# ============================================================
def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--image_dirs", nargs="+", required=True,
                   help="입력 이미지 폴더들 (예: Box/MSHF Box/OUWFD_downsize)")
    p.add_argument("--output_dir", type=str, required=True)

    # YOLO
    p.add_argument("--yolo_weights", type=str, required=True)
    p.add_argument("--yolo_device", type=str, default="0")
    p.add_argument("--yolo_img_size", type=int, default=640)

    # Seg
    p.add_argument("--seg_ckpt", type=str, required=True)
    p.add_argument("--seg_device", type=str, default="0")
    p.add_argument("--seg_input_size", nargs="+", type=int, default=[320, 320])
    p.add_argument("--seg_thr", type=float, default=0.5)
    p.add_argument("--seg_alpha", type=float, default=0.3)

    # Eval
    p.add_argument("--do_eval", action="store_true",
                   help="GT 기반 seg/det 평가 수행")
    p.add_argument("--gt_seg_dir", type=str, default="/mnt/richul_FM/UWF_seg_det/datasets/Seg/others")

    # metadata (원래 inference CSV 컬럼 채우기용)
    p.add_argument("--meta_data", type=str, default=None)
    p.add_argument("--dataset_name", type=str, default=None, choices=[None, "mshf", "ouwfd"])

    return p.parse_args()


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- seg device
    os.environ["CUDA_VISIBLE_DEVICES"] = args.seg_device
    seg_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seg_model = load_seg_model(args.seg_ckpt, seg_device)

    if len(args.seg_input_size) == 1:
        seg_input_size = int(args.seg_input_size[0])
    else:
        seg_input_size = (args.seg_input_size[0], args.seg_input_size[1])
    seg_tf = build_seg_transform(seg_input_size)

    # ---- yolo
    yolo_model, yolo_device, yolo_stride, yolo_names = load_yolo_model(
        weights_path=args.yolo_weights,
        device_str=args.yolo_device,
        imgsz=args.yolo_img_size,
    )

    # ---- metadata (원래 코드 방식 복원)
    meta_dict = {}
    if args.meta_data is not None:
        if args.dataset_name == "mshf":
            meta_df = load_mshf_meta(args.meta_data)
            meta_dict = meta_df.set_index("stem")[["ILL", "CLA", "CON", "OQ"]].to_dict(orient="index")
        elif args.dataset_name == "ouwfd":
            meta_df = load_ouwfd_meta(args.meta_data)
            meta_dict = meta_df.set_index("stem")[["FOV", "ILL", "CON", "ART", "OQ"]].to_dict(orient="index")
        print(f"[INFO] Loaded metadata for {len(meta_dict)} entries from {args.meta_data}")
    else:
        print("[INFO] No meta_data provided. All meta columns will be empty.")

    # ---- image list
    image_paths: List[Path] = []
    for d in args.image_dirs:
        dpath = Path(d)
        # >>> CHANGED: suffix.lower() == ".jpg"
        image_paths.extend([p for p in dpath.rglob("*") if p.is_file() and p.suffix.lower() == ".jpg"])
    image_paths = sorted(image_paths)

    if not image_paths:
        print(f"[ERROR] no images found in: {args.image_dirs}")
        return

    print(f"[INFO] # of images : {len(image_paths)}")

    gt_seg_dir = Path(args.gt_seg_dir)

    # ---- fixed IoU threshold for AP50
    # >>> CHANGED: eval_iou_thr 고정 (AP50)
    EVAL_IOU_THR = 0.5

    rows = []
    eval_rows = []

    seg_dices = []
    seg_ious = []

    det_counts = {
        "disc": {"TP": 0, "FP": 0, "FN": 0},
        "macula": {"TP": 0, "FP": 0, "FN": 0},
    }

    pred_buf = {0: [], 1: []}
    gt_buf = {0: {}, 1: {}}

    for img_path in image_paths:
        print(f"[PROC] {img_path}")

        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            print(f"[WARN] missing: {img_path}")
            continue

        h, w = img_bgr.shape[:2]
        pil_img = Image.open(str(img_path)).convert("RGB")

        # --- Segmentation
        mask_bool = run_seg_on_image(seg_model, seg_device, pil_img, seg_tf, thr=args.seg_thr)
        roi_pixels = int(mask_bool.sum())
        total_pixels = int(h * w) if h > 0 and w > 0 else 1
        pixel_percent = 100.0 * roi_pixels / total_pixels

        # --- YOLO detection (top1)
        yolo_results = run_yolo_on_image(
            yolo_model, yolo_device, yolo_stride, img_bgr,
            conf_thres=0.25, iou_thres=0.45, imgsz=args.yolo_img_size
        )
        disc_found, disc_conf, disc_box = yolo_results.get(0, (0, 0.0, None))
        mac_found,  mac_conf,  mac_box  = yolo_results.get(1, (0, 0.0, None))

        # --- Metadata 매칭 (원래 코드 방식)
        stem = img_path.stem
        meta = meta_dict.get(stem, {})

        if args.dataset_name == "mshf":
            ILL = meta.get("ILL", "")
            CLA = meta.get("CLA", "")
            CON = meta.get("CON", "")
            OQ  = meta.get("OQ", "")
            Diagnosis = ""
            FOV = ""
            ART = ""
        elif args.dataset_name == "ouwfd":
            FOV = meta.get("FOV", "")
            ART = meta.get("ART", "")
            ILL = meta.get("ILL", "")
            CON = meta.get("CON", "")
            OQ  = meta.get("OQ", "")
            Diagnosis = ""
            CLA = ""
        else:
            Diagnosis = FOV = ILL = CLA = CON = ART = OQ = ""

        # -------------------- EVAL --------------------
        seg_dice = seg_iou = ""
        disc_tp = disc_fp = disc_fn = ""
        mac_tp = mac_fp = mac_fn = ""
        disc_best_iou = mac_best_iou = ""

        # overlay에 사용할 macula GT 박스
        macula_gt_boxes = None

        if args.do_eval:
            img_key = img_path.name

            # GT txt 1회만 읽고 재사용 (중복 방지)
            gt_txt_path = img_path.with_suffix(".txt")
            gt_boxes_by_cls = yolo_txt_to_boxes_xyxy(gt_txt_path, w, h) if gt_txt_path.exists() else {}

            # overlay에 사용할 macula GT boxes (class=1)
            macula_gt_boxes = gt_boxes_by_cls.get(1, [])

            # GT buffer (mAP용)
            for cid in [0, 1]:
                gt_buf[cid][img_key] = gt_boxes_by_cls.get(cid, [])

            # Pred buffer (mAP용, 전체 박스)
            det_all = run_yolo_on_image_all(
                yolo_model, yolo_device, yolo_stride, img_bgr,
                conf_thres=0.001, iou_thres=0.45, imgsz=args.yolo_img_size
            )
            for cid, conf, box in det_all:
                if cid in (0, 1):
                    pred_buf[cid].append((img_key, conf, box))

            # Seg GT
            gt_mask_path = find_seg_gt_mask(img_path, gt_seg_dir)
            if gt_mask_path is not None:
                gt_mask = load_gt_mask_as_bool(gt_mask_path)
                if gt_mask.shape[:2] != mask_bool.shape[:2]:
                    gt_mask = cv2.resize(
                        gt_mask.astype(np.uint8),
                        (mask_bool.shape[1], mask_bool.shape[0]),
                        interpolation=cv2.INTER_NEAREST
                    ).astype(bool)
                seg_dice, seg_iou = dice_iou(mask_bool, gt_mask)
                seg_dices.append(seg_dice)
                seg_ious.append(seg_iou)

            # top1 det eval (AP50 IoU_thr=0.5로 평가)
            d_tp, d_fp, d_fn, d_best = eval_top1_det(int(disc_found), disc_box, gt_boxes_by_cls.get(0, []), EVAL_IOU_THR)
            m_tp, m_fp, m_fn, m_best = eval_top1_det(int(mac_found),  mac_box,  gt_boxes_by_cls.get(1, []), EVAL_IOU_THR)

            det_counts["disc"]["TP"] += d_tp
            det_counts["disc"]["FP"] += d_fp
            det_counts["disc"]["FN"] += d_fn
            det_counts["macula"]["TP"] += m_tp
            det_counts["macula"]["FP"] += m_fp
            det_counts["macula"]["FN"] += m_fn

            disc_tp, disc_fp, disc_fn, disc_best_iou = d_tp, d_fp, d_fn, d_best
            mac_tp, mac_fp, mac_fn, mac_best_iou = m_tp, m_fp, m_fn, m_best

        # --- Overlay save
        # >>> CHANGED: macula_gt_boxes 전달
        overlay = make_overlay(img_bgr, mask_bool, yolo_results, seg_alpha=args.seg_alpha, macula_gt_boxes=macula_gt_boxes)
        save_name = f"{stem}_px{pixel_percent:.2f}_disc{disc_found}_conf{disc_conf:.3f}_macula{mac_found}_conf{mac_conf:.3f}.jpg"
        save_path = output_dir / save_name
        cv2.imwrite(str(save_path), overlay)

        # --- inference row (원래 컬럼 채움)
        row = {
            "fname": img_path.name,
            "Diagnosis": Diagnosis,
            "FOV": FOV,
            "ILL": ILL,
            "CLA": CLA,
            "CON": CON,
            "ART": ART,
            "OQ": OQ,
            "PixelPercent": pixel_percent,
            "Macula YN": int(mac_found),
            "Macula Conf": float(mac_conf),
            "Disc YN": int(disc_found),
            "Disc Conf": float(disc_conf),
        }
        rows.append(row)

        if args.do_eval:
            eval_rows.append({
                "fname": img_path.name,
                "Seg Dice": seg_dice,
                "Seg IoU": seg_iou,
                "Disc TP": disc_tp,
                "Disc FP": disc_fp,
                "Disc FN": disc_fn,
                "Disc bestIoU": disc_best_iou,
                "Macula TP": mac_tp,
                "Macula FP": mac_fp,
                "Macula FN": mac_fn,
                "Macula bestIoU": mac_best_iou,
            })

    # -------------------- SAVE CSVs --------------------
    INFER_COLUMNS = [
        "fname", "Diagnosis", "FOV", "ILL", "CLA", "CON", "ART", "OQ",
        "PixelPercent", "Macula YN", "Macula Conf", "Disc YN", "Disc Conf"
    ]

    if rows:
        df_infer = pd.DataFrame(rows, columns=INFER_COLUMNS)
        csv_path = output_dir / "UWF_seg_det_inference_result.csv"
        df_infer.to_csv(csv_path, index=False)
        print(f"[INFO] Saved INFER CSV: {csv_path}")
    else:
        print("[WARN] No rows to save in inference CSV")

    EVAL_COLUMNS = [
        "fname",
        "Seg Dice", "Seg IoU",
        "Disc TP", "Disc FP", "Disc FN", "Disc bestIoU",
        "Macula TP", "Macula FP", "Macula FN", "Macula bestIoU",
    ]

    if args.do_eval and eval_rows:
        df_eval = pd.DataFrame(eval_rows, columns=EVAL_COLUMNS)
        eval_csv_path = output_dir / "UWF_seg_det_eval_per_image.csv"
        df_eval.to_csv(eval_csv_path, index=False)
        print(f"[INFO] Saved EVAL per-image CSV: {eval_csv_path}")

    if args.do_eval:
        # >>> CHANGED: AP50 only (iou_thr=0.5 고정)
        ap_disc, tp_disc, fp_disc = compute_ap_per_class(pred_buf[0], gt_buf[0], 0, iou_thr=EVAL_IOU_THR)
        ap_mac,  tp_mac,  fp_mac  = compute_ap_per_class(pred_buf[1], gt_buf[1], 1, iou_thr=EVAL_IOU_THR)
        mAP = (ap_disc + ap_mac) / 2.0

        mean_dice = float(np.mean(seg_dices)) if len(seg_dices) > 0 else np.nan
        mean_iou  = float(np.mean(seg_ious))  if len(seg_ious) > 0 else np.nan

        summary = [{
            "IoU_thr": EVAL_IOU_THR,
            "Seg mean Dice": mean_dice,
            "Seg mean IoU": mean_iou,
            "AP50 Disc": ap_disc,
            "AP50 Macula": ap_mac,
            "mAP50 (Disc+Macula)/2": mAP,
        }]
        df_sum = pd.DataFrame(summary)
        sum_path = output_dir / "UWF_seg_det_eval_summary.csv"
        df_sum.to_csv(sum_path, index=False)
        print(f"[INFO] Saved EVAL summary CSV (mAP50 포함): {sum_path}")

    print("[DONE]")


if __name__ == "__main__":
    main()
