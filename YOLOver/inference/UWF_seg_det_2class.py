
# output file name format: OCT_0001_px12.34_det1_conf0.876.png
import os
import sys
import random
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

# save DataFrame as csv
import pandas as pd


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
    return: {
                0: (found_disc,  disc_conf,  disc_box),
                1: (found_mac,   mac_conf,   mac_box),
            }
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
    
    # 기본 반환 구조 (아무 것도 못 찾았을 때)
    results = {
        0: (0, 0.0, None),  # disc
        1: (0, 0.0, None),  # macula
    }

    if det is None or len(det) == 0:
        return results

    # 각 class id (0: disc, 1: macula)에 대해 최고 conf 박스 1개씩 선택
    for class_id in [0, 1]:
        det_cls = det[det[:, 5] == class_id]
        if det_cls is None or len(det_cls) == 0:
            continue

        # 해당 클래스 중 가장 confidence 높은 것 선택
        best_idx = det_cls[:, 4].argmax()
        best_det = det_cls[best_idx].unsqueeze(0)  # [1,6]

        # 좌표 rescale
        best_det[:, :4] = scale_boxes(img_tensor.shape[2:], best_det[:, :4], img0.shape[:2]).round()

        x1, y1, x2, y2, conf, cls = best_det[0].tolist()
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        results[class_id] = (1, float(conf), (x1, y1, x2, y2))

    return results

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
    class_names = {
        0: "disc",
        1: "macula",
    }

    class_colors = {
        0: (0, 0, 255),   # disc = 빨강
        1: (0, 255, 0),   # macula = 초록
    }

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
        cv2.putText(overlay, label, (x1 + 2, y_text - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, lineType=cv2.LINE_AA)

    return overlay


# ============================================================
# MSHF 메타데이터 로딩 (3명 annotator → 다수결)
# ============================================================
def load_mshf_meta(xlsx_path: str) -> pd.DataFrame:
    """
    MSHF Individual_scores.xlsx 를 읽어서
    fname, stem, ILL, CLA, CON, OQ 가 있는 DataFrame 반환.

    - 엑셀은 2행 헤더 구조 (header=[0,1])
    - annotator 1/2/3의 0/1 점수를 다수결(>=2 → 1, 아니면 0)로 합침
    """
    # 2줄짜리 헤더
    df = pd.read_excel(xlsx_path, header=[0, 1])

    # 첫 번째 컬럼을 image name 으로 사용 
    fname = df.iloc[:, 0].astype(str)

    # annotator 열 자동 탐색
    # 예: ('annotator 1 Y', 'illumination') 이런 식의 multiindex
    ill_cols = [col for col in df.columns
                if 'annotator' in str(col[0]) and str(col[1]).lower() == 'illumination']
    cla_cols = [col for col in df.columns
                if 'annotator' in str(col[0]) and str(col[1]).lower() == 'clarity']
    con_cols = [col for col in df.columns
                if 'annotator' in str(col[0]) and str(col[1]).lower() == 'contrast']
    oq_cols  = [col for col in df.columns
                if 'annotator' in str(col[0]) and str(col[1]).lower() == 'overall']

    def majority_vote(cols):
        # 3명 annotator 열만 뽑아서 합산 후 2 이상이면 1, 아니면 0
        votes = df[cols].astype(float)
        return (votes.sum(axis=1) >= 2).astype(int)

    ILL = majority_vote(ill_cols)
    CLA = majority_vote(cla_cols)
    CON = majority_vote(con_cols)
    OQ  = majority_vote(oq_cols)

    meta = pd.DataFrame({
        "fname": fname,
        "ILL": ILL,
        "CLA": CLA,
        "CON": CON,
        "OQ":  OQ,
    })

    # stem (확장자 제거) → 이미지 파일과 매칭에 사용
    meta["stem"] = meta["fname"].apply(lambda x: Path(x).stem)

    return meta

# ============================================================
# OUWFD 메타데이터 (Ground Truth.xlsx) 로딩
# ============================================================
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
# 메인 루프
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--image_dir", type=str, required=True)
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

    # medadata
    parser.add_argument("--meta_data", type=str, default=None)
    parser.add_argument("--dataset_name", type=str, default=None)

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


    # ---------------- 메타데이터 Xlsx 로딩 ---------------- 
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


    # ---------------- 이미지 리스트 ----------------
    exts = [".jpg"]
    image_paths = sorted(
        [p for p in image_dir.rglob("*") if p.suffix.lower() in exts]
    )
    
    if not image_paths:
        print(f"[ERROR] no images contained in {image_dir}.")
        return

    print(f"[INFO] # of images : {len(image_paths)}")

    # if len(image_paths) > 30:
    #     image_paths = random.sample(image_paths, 30)
    #     print(f"[INFO] process only sample images : {len(image_paths)}")

    rows = []
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
        yolo_results = run_yolo_on_image(
            yolo_model,
            yolo_device,
            yolo_stride,
            img_bgr,
            conf_thres=0.25,
            iou_thres=0.45,
            imgsz=args.yolo_img_size
        )

        # --- Metadata 매칭 ----
        stem = img_path.stem
        meta = meta_dict.get(stem, {})

        if args.dataset_name == "mshf":
            ILL = meta.get("ILL", "")
            CLA = meta.get("CLA", "")
            CON = meta.get("CON", "")
            OQ  = meta.get("OQ", "")
            # MSHF Dataset : Diagnosis, FOV, ART X
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
            FOV = ART = ILL = CLA = CON = OQ = Diagnosis = ""
            

        # --- Overlay 생성 ---
        overlay = make_overlay(
            orig_bgr=img_bgr,
            mask_bool=mask_bool,
            yolo_info=yolo_results,
            seg_alpha=args.seg_alpha
        )

        # --- 파일 이름 구성 및 저장 ---
        # stem = img_path.stem
        disc_found, disc_conf, _ = yolo_results.get(0, (0, 0.0, None))
        macula_found, macula_conf, _ = yolo_results.get(1, (0, 0.0, None))

        save_name = f"{stem}_px{pixel_percent:.2f}_disc{disc_found}_conf{disc_conf:.3f}_macula{macula_found}_conf{macula_conf:.3f}.jpg"
        save_path = output_dir / save_name

        cv2.imwrite(str(save_path), overlay)
        print(f"  -> saved: {save_path}")

        # --- csv row 누적 ---
        rows.append({
            "fname" : img_path.name,
            "Diagnosis" : Diagnosis,
            "FOV" : FOV,
            "ILL" : ILL,
            "CLA" : CLA,
            "CON" : CON,
            "ART" : ART,
            "OQ"  : OQ,
            "PixelPercent" : pixel_percent,
            "Macula YN" : int(macula_found),
            "Macula Conf" : float(macula_conf),
            "Disc YN" : int(disc_found),
            "Disc Conf" : float(disc_conf),
        })

    if rows:
        df = pd.DataFrame(rows, columns = ["fname","Diagnosis","FOV","ILL",
            "CLA","CON","ART","OQ","PixelPercent","Macula YN","Macula Conf",
            "Disc YN","Disc Conf"])
        csv_path = output_dir / "UWF_seg_det_inference_result.csv"
        df.to_csv(csv_path, index=False)
        print(f"[INFO] Saved CSV : {csv_path}")
    else:
        print(f"[WARN] No rows to save in CSV")


    print("[DONE]")


if __name__ == "__main__":
    main()
