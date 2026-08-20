#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CSV의 image_path를 직접 읽어 다음 QC를 수행합니다.

1) Retinal ROI segmentation
2) Optic disc / fovea landmark detection (GitHub inference_tta_mean.py와 동일한 핵심 방식)
3) 원본 CSV 열을 유지하면서 QC 열 추가
4) 이미지 파일은 저장하지 않고 CSV만 저장

Output:
- uwf_qc_results.csv
- uwf_qc_summary.csv
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from tqdm import tqdm


# ---------------------------------------------------------------------
# Checkpoint / model utilities
# ---------------------------------------------------------------------

def extract_state_dict(checkpoint: Any) -> Dict[str, torch.Tensor]:
    """다양한 checkpoint 저장 형식에서 state_dict를 추출합니다."""
    if isinstance(checkpoint, dict):
        for key in (
            "state_dict",
            "model_state_dict",
            "model",
            "net",
            "network",
            "weights",
        ):
            value = checkpoint.get(key)
            if isinstance(value, dict) and value:
                checkpoint = value
                break

    if not isinstance(checkpoint, dict):
        raise TypeError(
            f"Unsupported checkpoint type: {type(checkpoint).__name__}"
        )

    state_dict: Dict[str, torch.Tensor] = {}
    for key, value in checkpoint.items():
        if not torch.is_tensor(value):
            continue

        new_key = str(key)
        # 흔한 wrapper prefix를 반복 제거합니다.
        changed = True
        while changed:
            changed = False
            for prefix in ("module.", "model.", "net.", "network."):
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix):]
                    changed = True

        state_dict[new_key] = value

    if not state_dict:
        raise RuntimeError("No tensor entries found in checkpoint state_dict.")

    return state_dict


def infer_output_channels(
    state_dict: Dict[str, torch.Tensor],
    fallback: int,
) -> int:
    """
    SMP segmentation head의 마지막 convolution weight에서 출력 channel 수를 추정합니다.
    """
    candidate_keys: List[str] = []

    for key, value in state_dict.items():
        if (
            value.ndim == 4
            and "segmentation_head" in key
            and key.endswith("weight")
        ):
            candidate_keys.append(key)

    if candidate_keys:
        # 일반적으로 segmentation_head.0.weight가 최종 출력 conv입니다.
        key = sorted(candidate_keys)[-1]
        return int(state_dict[key].shape[0])

    # generic fallback: 마지막 1x1 conv 후보
    conv_candidates: List[Tuple[str, torch.Tensor]] = []
    for key, value in state_dict.items():
        if value.ndim == 4 and tuple(value.shape[-2:]) == (1, 1):
            conv_candidates.append((key, value))

    if conv_candidates:
        key, value = conv_candidates[-1]
        out_ch = int(value.shape[0])
        if 1 <= out_ch <= 32:
            return out_ch

    return int(fallback)


def parse_encoder_candidates(
    encoder_hint: str,
    extra_candidates: str,
) -> List[str]:
    values: List[str] = []

    if encoder_hint and encoder_hint.lower() != "auto":
        values.append(encoder_hint)

    for value in extra_candidates.split(","):
        value = value.strip()
        if value:
            values.append(value)

    defaults = [
        "resnet50",
        "resnet34",
        "efficientnet-b0",
        "timm-efficientnet-b0",
        "mobilenet_v2",
    ]
    values.extend(defaults)

    # 순서 보존 중복 제거
    return list(dict.fromkeys(values))


def build_smp_model(
    architecture: str,
    encoder_name: str,
    classes: int,
) -> torch.nn.Module:
    """요청한 segmentation_models_pytorch 모델을 생성합니다."""
    architecture_key = architecture.strip().lower().replace("-", "").replace("_", "")

    common_kwargs = {
        "encoder_name": encoder_name,
        "encoder_weights": None,
        "in_channels": 3,
        "classes": classes,
        "activation": None,
    }

    if architecture_key == "unet":
        return smp.Unet(**common_kwargs)

    if architecture_key in {"deeplabv3", "dlv3"}:
        return smp.DeepLabV3(**common_kwargs)

    if architecture_key in {
        "deeplabv3plus",
        "deeplabv3+",
        "dlv3plus",
        "dlv3+",
    }:
        return smp.DeepLabV3Plus(**common_kwargs)

    raise ValueError(
        f"Unsupported SMP architecture: {architecture}. "
        "Supported: unet, deeplabv3, deeplabv3plus"
    )


def load_smp_model(
    checkpoint_path: Path,
    device: torch.device,
    architecture: str,
    encoder_hint: str,
    encoder_candidates: str,
    fallback_classes: int,
    model_name: str,
) -> Tuple[torch.nn.Module, str, int]:
    """
    checkpoint와 정확히 일치하는 SMP 모델을 strict=True로 로드합니다.

    Landmark:
      smp.Unet + resnet50

    ROI:
      smp.DeepLabV3Plus + resnet50
    """
    raw = torch.load(str(checkpoint_path), map_location="cpu")
    state_dict = extract_state_dict(raw)
    classes = infer_output_channels(state_dict, fallback=fallback_classes)

    errors: List[str] = []

    for encoder in parse_encoder_candidates(
        encoder_hint=encoder_hint,
        extra_candidates=encoder_candidates,
    ):
        try:
            model = build_smp_model(
                architecture=architecture,
                encoder_name=encoder,
                classes=classes,
            )
            model.load_state_dict(state_dict, strict=True)
            model.to(device).eval()

            print(
                f"[{model_name}] loaded | "
                f"architecture={architecture} | "
                f"encoder={encoder} | classes={classes} | "
                f"checkpoint={checkpoint_path}"
            )
            return model, encoder, classes

        except Exception as exc:
            errors.append(f"{encoder}: {type(exc).__name__}: {exc}")

    short_errors = "\n".join(f"  - {x}" for x in errors)
    raise RuntimeError(
        f"Could not load {model_name} checkpoint.\n"
        f"Expected architecture: smp.{architecture}\n"
        f"Checkpoint: {checkpoint_path}\n"
        f"Tried encoders:\n{short_errors}"
    )


def find_roi_checkpoint(
    repo_dir: Path,
    landmark_checkpoint: Path,
    explicit_roi_checkpoint: Optional[str],
) -> Path:
    if explicit_roi_checkpoint:
        path = Path(explicit_roi_checkpoint).expanduser()
        if not path.is_absolute():
            path = (repo_dir / path).resolve()
        return path

    weights_dir = repo_dir / "weights"
    all_weights = sorted(
        [
            p
            for p in weights_dir.iterdir()
            if p.is_file() and p.suffix.lower() in {".pth", ".pt", ".ckpt"}
        ]
    )

    landmark_resolved = landmark_checkpoint.resolve()
    others = [p for p in all_weights if p.resolve() != landmark_resolved]

    named = [
        p
        for p in others
        if re.search(r"(roi|seg|segment|mask|retina)", p.name, flags=re.I)
    ]

    if len(named) == 1:
        print(f"[ROI] auto-detected checkpoint: {named[0]}")
        return named[0]

    if len(named) > 1:
        options = "\n".join(f"  - {p}" for p in named)
        raise RuntimeError(
            "Multiple ROI checkpoint candidates were found.\n"
            "Set ROI_CKPT explicitly in run_csv_qc.sh:\n"
            f"{options}"
        )

    if len(others) == 1:
        print(f"[ROI] using the only non-landmark checkpoint: {others[0]}")
        return others[0]

    options = "\n".join(f"  - {p}" for p in others) or "  (none)"
    raise RuntimeError(
        "ROI checkpoint could not be identified automatically.\n"
        "Set ROI_CKPT explicitly in run_csv_qc.sh.\n"
        f"Other weight files:\n{options}"
    )


# ---------------------------------------------------------------------
# Image / mask utilities
# ---------------------------------------------------------------------

def resolve_image_path(
    value: Any,
    csv_path: Path,
    image_root: Optional[Path],
) -> Path:
    raw = str(value).strip()
    path = Path(raw).expanduser()

    candidates: List[Path] = []

    if path.is_absolute():
        candidates.append(path)
    else:
        if image_root is not None:
            candidates.append(image_root / path)
        candidates.append(csv_path.parent / path)
        candidates.append(path)

    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate.exists():
            return candidate

    # 실패하더라도 가장 가능성 높은 경로를 반환하여 CSV에 기록
    if candidates:
        return candidates[0].resolve()
    return path.resolve()


def largest_component(
    binary_mask: np.ndarray,
) -> Tuple[np.ndarray, Optional[Tuple[float, float]], int]:
    mask = (binary_mask > 0).astype(np.uint8)

    if mask.max() == 0:
        return np.zeros_like(mask), None, 0

    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8,
    )

    if count <= 1:
        return np.zeros_like(mask), None, 0

    areas = stats[1:, cv2.CC_STAT_AREA]
    selected_label = 1 + int(np.argmax(areas))
    selected = (labels == selected_label).astype(np.uint8)
    area = int(stats[selected_label, cv2.CC_STAT_AREA])
    cx, cy = centroids[selected_label]

    return selected, (float(cx), float(cy)), area


def map_center_to_original(
    center_xy: Optional[Tuple[float, float]],
    original_width: int,
    original_height: int,
    resized_size: int,
) -> Optional[Tuple[int, int]]:
    if center_xy is None:
        return None

    x, y = center_xy
    mapped_x = int(round(x * original_width / float(resized_size)))
    mapped_y = int(round(y * original_height / float(resized_size)))
    return mapped_x, mapped_y


def center_xy_columns(
    center: Optional[Tuple[int, int]],
) -> Tuple[Optional[int], Optional[int], str]:
    if center is None:
        return None, None, ""

    x, y = center
    return int(x), int(y), f"{int(x)},{int(y)}"


def mean_pairwise_distance(
    points: Sequence[Tuple[int, int]],
) -> float:
    if len(points) <= 1:
        return 0.0

    distances: List[float] = []
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            distances.append(
                float(
                    math.hypot(
                        points[i][0] - points[j][0],
                        points[i][1] - points[j][1],
                    )
                )
            )

    return float(np.mean(distances)) if distances else 0.0


def find_blob_peak(
    heatmap: np.ndarray,
    threshold: float,
) -> Optional[Tuple[int, int, float]]:
    """
    GitHub inference.py의 핵심 방식:
    threshold → connected components → global max 포함 component 선택
    """
    binary = (heatmap >= threshold).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8,
    )

    if count <= 1:
        return None

    global_y, global_x = np.unravel_index(
        int(np.argmax(heatmap)),
        heatmap.shape,
    )
    selected_label = int(labels[global_y, global_x])

    if selected_label == 0:
        areas = stats[1:, cv2.CC_STAT_AREA]
        selected_label = 1 + int(np.argmax(areas))

    component = labels == selected_label
    masked = heatmap * component.astype(heatmap.dtype)
    y, x = np.unravel_index(int(np.argmax(masked)), heatmap.shape)

    return int(x), int(y), float(heatmap[y, x])


# ---------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------

def infer_with_flip_tta(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    use_tta: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return:
      predictions: (T,C,H,W), 원 좌표계로 flip 복원된 prediction
      mean_map:    (C,H,W)
    """
    predictions: List[np.ndarray] = []

    output = torch.sigmoid(model(image_tensor))[0]
    predictions.append(
        output.detach().cpu().numpy().astype(np.float32)
    )

    if use_tta:
        horizontal_input = torch.flip(image_tensor, dims=[3])
        horizontal_output = torch.sigmoid(
            model(horizontal_input)
        )[0]
        horizontal_output = torch.flip(horizontal_output, dims=[2])
        predictions.append(
            horizontal_output.detach().cpu().numpy().astype(np.float32)
        )

        vertical_input = torch.flip(image_tensor, dims=[2])
        vertical_output = torch.sigmoid(
            model(vertical_input)
        )[0]
        vertical_output = torch.flip(vertical_output, dims=[1])
        predictions.append(
            vertical_output.detach().cpu().numpy().astype(np.float32)
        )

    stacked = np.stack(predictions, axis=0).astype(np.float32)
    mean_map = stacked.mean(axis=0).astype(np.float32)

    return stacked, mean_map


def infer_roi(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    output_channels: int,
    threshold: float,
    foreground_channel: int,
    use_tta: bool,
    return_mask: bool = False,
) -> Dict[str, Any]:
    """
    ROI segmentation 결과를 계산합니다.
    - 1 channel: sigmoid
    - >=2 channels: softmax 후 foreground_channel 사용
    """
    logits_list: List[torch.Tensor] = []

    logits_list.append(model(image_tensor)[0])

    if use_tta:
        horizontal_input = torch.flip(image_tensor, dims=[3])
        horizontal_logits = model(horizontal_input)[0]
        horizontal_logits = torch.flip(horizontal_logits, dims=[2])
        logits_list.append(horizontal_logits)

        vertical_input = torch.flip(image_tensor, dims=[2])
        vertical_logits = model(vertical_input)[0]
        vertical_logits = torch.flip(vertical_logits, dims=[1])
        logits_list.append(vertical_logits)

    mean_logits = torch.stack(logits_list, dim=0).mean(dim=0)

    if output_channels == 1:
        probability = torch.sigmoid(mean_logits[0])
        selected_channel = 0
    else:
        if not (0 <= foreground_channel < output_channels):
            raise ValueError(
                f"ROI foreground channel {foreground_channel} is invalid "
                f"for output_channels={output_channels}"
            )
        probability = torch.softmax(mean_logits, dim=0)[foreground_channel]
        selected_channel = foreground_channel

    probability_np = (
        probability.detach().cpu().numpy().astype(np.float32)
    )
    binary = (probability_np >= threshold).astype(np.uint8)
    selected_mask, _, area = largest_component(binary)

    total_pixels = int(selected_mask.size)
    area_ratio = (
        float(area / total_pixels)
        if total_pixels > 0
        else float("nan")
    )

    if area > 0:
        values = probability_np[selected_mask > 0]
        mean_confidence = float(values.mean())
        max_confidence = float(values.max())
    else:
        mean_confidence = 0.0
        max_confidence = float(probability_np.max())

    result: Dict[str, Any] = {
        "roi_detected": bool(area > 0),
        "roi_area_pixels_512": int(area),
        "roi_area_ratio": area_ratio,
        "roi_mean_confidence": mean_confidence,
        "roi_max_confidence": max_confidence,
        "roi_foreground_channel": int(selected_channel),
    }

    if return_mask:
        result["_roi_mask_512"] = selected_mask

    return result


def infer_landmarks(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    image_size: int,
    original_width: int,
    original_height: int,
    threshold: float,
    blob_threshold: float,
    use_tta: bool,
    confidence_threshold: float,
    spread_threshold: float,
) -> Dict[str, Any]:
    predictions, mean_map = infer_with_flip_tta(
        model=model,
        image_tensor=image_tensor,
        use_tta=use_tta,
    )

    if mean_map.shape[0] < 2:
        raise RuntimeError(
            f"Landmark model output has {mean_map.shape[0]} channels; "
            "at least 2 channels are required."
        )

    disc_tta_points: List[Tuple[int, int]] = []
    fovea_tta_points: List[Tuple[int, int]] = []
    disc_tta_values: List[float] = []
    fovea_tta_values: List[float] = []

    for prediction in predictions:
        disc_peak = find_blob_peak(prediction[0], blob_threshold)
        fovea_peak = find_blob_peak(prediction[1], blob_threshold)

        if disc_peak is not None:
            x, y, value = disc_peak
            disc_tta_points.append((x, y))
            disc_tta_values.append(value)

        if fovea_peak is not None:
            x, y, value = fovea_peak
            fovea_tta_points.append((x, y))
            fovea_tta_values.append(value)

    disc_mean = mean_map[0]
    fovea_mean = mean_map[1]

    disc_binary = (disc_mean > threshold).astype(np.uint8)
    fovea_binary = (fovea_mean > threshold).astype(np.uint8)

    disc_component, disc_center_512, disc_area = largest_component(
        disc_binary
    )
    fovea_component, fovea_center_512, fovea_area = largest_component(
        fovea_binary
    )

    disc_detected = disc_center_512 is not None
    fovea_detected = fovea_center_512 is not None

    disc_center_original = map_center_to_original(
        disc_center_512,
        original_width=original_width,
        original_height=original_height,
        resized_size=image_size,
    )
    fovea_center_original = map_center_to_original(
        fovea_center_512,
        original_width=original_width,
        original_height=original_height,
        resized_size=image_size,
    )

    disc_x, disc_y, disc_center_text = center_xy_columns(
        disc_center_original
    )
    fovea_x, fovea_y, fovea_center_text = center_xy_columns(
        fovea_center_original
    )

    if disc_area > 0:
        disc_final_values = disc_mean[disc_component > 0]
        disc_confidence = float(disc_final_values.max())
    else:
        disc_confidence = float(disc_mean.max())

    if fovea_area > 0:
        fovea_final_values = fovea_mean[fovea_component > 0]
        fovea_confidence = float(fovea_final_values.max())
    else:
        fovea_confidence = float(fovea_mean.max())

    disc_tta_mean_confidence = (
        float(np.mean(disc_tta_values))
        if disc_tta_values
        else 0.0
    )
    fovea_tta_mean_confidence = (
        float(np.mean(fovea_tta_values))
        if fovea_tta_values
        else 0.0
    )

    disc_tta_mean_distance = mean_pairwise_distance(
        disc_tta_points
    )
    fovea_tta_mean_distance = mean_pairwise_distance(
        fovea_tta_points
    )

    def status(
        detected: bool,
        mean_confidence: float,
        spread: float,
    ) -> str:
        if not detected:
            return "Not detected"
        if mean_confidence < confidence_threshold:
            return "Low confidence"
        if spread > spread_threshold:
            return "Unstable"
        return "Good"

    return {
        "disc_detected": bool(disc_detected),
        "fovea_detected": bool(fovea_detected),
        "both_disc_fovea_detected": bool(
            disc_detected and fovea_detected
        ),
        "neither_disc_fovea_detected": bool(
            (not disc_detected) and (not fovea_detected)
        ),
        "disc_x": disc_x,
        "disc_y": disc_y,
        "disc_center": disc_center_text,
        "fovea_x": fovea_x,
        "fovea_y": fovea_y,
        "fovea_center": fovea_center_text,
        "disc_confidence": disc_confidence,
        "fovea_confidence": fovea_confidence,
        "disc_tta_mean_confidence": disc_tta_mean_confidence,
        "fovea_tta_mean_confidence": fovea_tta_mean_confidence,
        "disc_tta_mean_distance_512": disc_tta_mean_distance,
        "fovea_tta_mean_distance_512": fovea_tta_mean_distance,
        "disc_tta_detection_count": int(len(disc_tta_points)),
        "fovea_tta_detection_count": int(len(fovea_tta_points)),
        "disc_status": status(
            disc_detected,
            disc_tta_mean_confidence,
            disc_tta_mean_distance,
        ),
        "fovea_status": status(
            fovea_detected,
            fovea_tta_mean_confidence,
            fovea_tta_mean_distance,
        ),
    }



# ---------------------------------------------------------------------
# First-N visual QC output
# ---------------------------------------------------------------------

def safe_filename(value: Any, fallback: str) -> str:
    """파일명으로 사용할 수 없는 문자를 제거합니다."""
    name = Path(str(value)).stem.strip()
    name = re.sub(r"[^0-9A-Za-z._-]+", "_", name)
    name = name.strip("._")
    return name[:120] if name else fallback


def put_text_with_background(
    image: np.ndarray,
    text: str,
    origin: Tuple[int, int],
    font_scale: float = 0.62,
    thickness: int = 1,
) -> None:
    """가독성 있는 텍스트를 image에 직접 그립니다."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    x, y = origin
    (width, height), baseline = cv2.getTextSize(
        text,
        font,
        font_scale,
        thickness,
    )
    x2 = min(image.shape[1] - 1, x + width + 10)
    y1 = max(0, y - height - 8)
    y2 = min(image.shape[0] - 1, y + baseline + 5)

    cv2.rectangle(image, (x, y1), (x2, y2), (0, 0, 0), -1)
    cv2.putText(
        image,
        text,
        (x + 5, y - 3),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def make_qc_visual(
    original_bgr: np.ndarray,
    roi_mask_512: np.ndarray,
    roi_result: Dict[str, Any],
    landmark_result: Dict[str, Any],
    display_name: str,
    max_panel_width: int = 900,
) -> np.ndarray:
    """
    원본과 QC overlay를 좌우로 연결한 결과 이미지를 생성합니다.

    표시:
    - ROI: cyan 반투명 영역 + contour
    - Disc: yellow circle
    - Fovea: magenta circle
    """
    original = original_bgr.copy()
    overlay = original_bgr.copy()

    height, width = overlay.shape[:2]

    roi_mask = cv2.resize(
        (roi_mask_512 > 0).astype(np.uint8),
        (width, height),
        interpolation=cv2.INTER_NEAREST,
    )

    if roi_mask.any():
        color_layer = np.zeros_like(overlay)
        color_layer[roi_mask > 0] = (255, 220, 0)  # BGR cyan
        overlay = cv2.addWeighted(overlay, 1.0, color_layer, 0.24, 0)

        contours, _ = cv2.findContours(
            roi_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        cv2.drawContours(
            overlay,
            contours,
            -1,
            (255, 255, 0),
            max(2, round(min(width, height) / 350)),
        )

    radius = max(7, round(min(width, height) / 55))
    line_width = max(2, round(min(width, height) / 350))

    if bool(landmark_result.get("disc_detected", False)):
        disc_x = int(landmark_result["disc_x"])
        disc_y = int(landmark_result["disc_y"])
        cv2.circle(
            overlay,
            (disc_x, disc_y),
            radius,
            (0, 255, 255),
            line_width,
        )
        cv2.putText(
            overlay,
            "DISC",
            (disc_x + radius + 4, max(20, disc_y)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

    if bool(landmark_result.get("fovea_detected", False)):
        fovea_x = int(landmark_result["fovea_x"])
        fovea_y = int(landmark_result["fovea_y"])
        cv2.circle(
            overlay,
            (fovea_x, fovea_y),
            radius,
            (255, 0, 255),
            line_width,
        )
        cv2.putText(
            overlay,
            "FOVEA",
            (fovea_x + radius + 4, max(20, fovea_y)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 0, 255),
            2,
            cv2.LINE_AA,
        )

    roi_ratio = float(roi_result.get("roi_area_ratio", float("nan")))
    disc_conf = float(
        landmark_result.get("disc_confidence", float("nan"))
    )
    fovea_conf = float(
        landmark_result.get("fovea_confidence", float("nan"))
    )

    info_lines = [
        display_name,
        (
            f"ROI ratio={roi_ratio:.4f} | "
            f"ROI conf={float(roi_result.get('roi_mean_confidence', float('nan'))):.3f}"
        ),
        (
            f"Disc={bool(landmark_result.get('disc_detected', False))} "
            f"(conf={disc_conf:.3f}) | "
            f"Fovea={bool(landmark_result.get('fovea_detected', False))} "
            f"(conf={fovea_conf:.3f})"
        ),
    ]

    for i, line in enumerate(info_lines):
        put_text_with_background(
            overlay,
            line,
            (8, 28 + i * 30),
            font_scale=0.58,
            thickness=1,
        )

    # 너무 큰 원본은 패널당 최대 너비에 맞춰 줄입니다.
    scale = min(1.0, max_panel_width / float(width))
    if scale < 1.0:
        new_size = (
            max(1, int(round(width * scale))),
            max(1, int(round(height * scale))),
        )
        original = cv2.resize(
            original,
            new_size,
            interpolation=cv2.INTER_AREA,
        )
        overlay = cv2.resize(
            overlay,
            new_size,
            interpolation=cv2.INTER_AREA,
        )

    divider = np.full(
        (original.shape[0], 8, 3),
        255,
        dtype=np.uint8,
    )
    return np.concatenate([original, divider, overlay], axis=1)


def save_contact_sheet(
    image_paths: Sequence[Path],
    output_path: Path,
    columns: int = 2,
    tile_width: int = 1100,
) -> None:
    """저장된 QC 이미지들을 한 장의 contact sheet로 합칩니다."""
    loaded: List[np.ndarray] = []

    for path in image_paths:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            continue

        scale = tile_width / float(image.shape[1])
        tile_height = max(1, int(round(image.shape[0] * scale)))
        resized = cv2.resize(
            image,
            (tile_width, tile_height),
            interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR,
        )
        loaded.append(resized)

    if not loaded:
        return

    tile_height = max(image.shape[0] for image in loaded)
    rows = math.ceil(len(loaded) / columns)
    canvas = np.full(
        (
            rows * tile_height + (rows - 1) * 10,
            columns * tile_width + (columns - 1) * 10,
            3,
        ),
        245,
        dtype=np.uint8,
    )

    for index, image in enumerate(loaded):
        row = index // columns
        column = index % columns
        y = row * (tile_height + 10)
        x = column * (tile_width + 10)
        canvas[y:y + image.shape[0], x:x + image.shape[1]] = image

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), canvas)


# ---------------------------------------------------------------------
# CSV / summary
# ---------------------------------------------------------------------

def select_rows(
    dataframe: pd.DataFrame,
    image_column: str,
    sample_n: int,
    sample_mode: str,
    random_seed: int,
) -> pd.DataFrame:
    selected = dataframe[
        dataframe[image_column].notna()
        & dataframe[image_column].astype(str).str.strip().ne("")
    ].copy()

    selected.insert(
        0,
        "qc_source_index",
        selected.index.astype(str),
    )

    if sample_n > 0 and len(selected) > sample_n:
        if sample_mode == "first":
            selected = selected.head(sample_n).copy()
        else:
            selected = selected.sample(
                n=sample_n,
                random_state=random_seed,
            ).copy()

    return selected.reset_index(drop=True)


def safe_ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return float("nan")
    return float(numerator / denominator)


def build_summary(
    result_dataframe: pd.DataFrame,
    csv_path: Path,
    output_path: Path,
    roi_checkpoint: Path,
    landmark_checkpoint: Path,
    roi_encoder: str,
    landmark_encoder: str,
) -> pd.DataFrame:
    total = int(len(result_dataframe))
    success_mask = result_dataframe["qc_status"].eq("OK")
    successful = int(success_mask.sum())
    failed = total - successful

    ok = result_dataframe.loc[success_mask].copy()

    both_count = int(
        ok["both_disc_fovea_detected"].fillna(False).sum()
    )
    disc_not_count = int(
        (~ok["disc_detected"].fillna(False)).sum()
    )
    fovea_not_count = int(
        (~ok["fovea_detected"].fillna(False)).sum()
    )
    neither_count = int(
        ok["neither_disc_fovea_detected"].fillna(False).sum()
    )
    roi_detected_count = int(
        ok["roi_detected"].fillna(False).sum()
    )

    roi_ratio = pd.to_numeric(
        ok["roi_area_ratio"],
        errors="coerce",
    )

    rows = [
        ("input_csv", str(csv_path), "", ""),
        ("output_results_csv", str(output_path), "", ""),
        ("roi_checkpoint", str(roi_checkpoint), "", ""),
        ("landmark_checkpoint", str(landmark_checkpoint), "", ""),
        ("roi_encoder", roi_encoder, "", ""),
        ("landmark_encoder", landmark_encoder, "", ""),
        ("selected_rows", total, total, 1.0 if total else np.nan),
        (
            "successfully_processed",
            successful,
            total,
            safe_ratio(successful, total),
        ),
        (
            "failed",
            failed,
            total,
            safe_ratio(failed, total),
        ),
        (
            "roi_detected",
            roi_detected_count,
            successful,
            safe_ratio(roi_detected_count, successful),
        ),
        (
            "both_disc_fovea_detected",
            both_count,
            successful,
            safe_ratio(both_count, successful),
        ),
        (
            "disc_not_detected",
            disc_not_count,
            successful,
            safe_ratio(disc_not_count, successful),
        ),
        (
            "fovea_not_detected",
            fovea_not_count,
            successful,
            safe_ratio(fovea_not_count, successful),
        ),
        (
            "neither_disc_fovea_detected",
            neither_count,
            successful,
            safe_ratio(neither_count, successful),
        ),
        (
            "roi_area_ratio_mean",
            float(roi_ratio.mean()) if roi_ratio.notna().any() else np.nan,
            successful,
            "",
        ),
        (
            "roi_area_ratio_median",
            float(roi_ratio.median()) if roi_ratio.notna().any() else np.nan,
            successful,
            "",
        ),
        (
            "roi_area_ratio_min",
            float(roi_ratio.min()) if roi_ratio.notna().any() else np.nan,
            successful,
            "",
        ),
        (
            "roi_area_ratio_max",
            float(roi_ratio.max()) if roi_ratio.notna().any() else np.nan,
            successful,
            "",
        ),
    ]

    summary = pd.DataFrame(
        rows,
        columns=["metric", "value", "denominator", "ratio"],
    )
    return summary


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--csv_path", required=True)
    parser.add_argument("--image_column", default="image_path")
    parser.add_argument("--image_root", default="")

    parser.add_argument("--repo_dir", required=True)
    parser.add_argument("--roi_ckpt", default="")
    parser.add_argument("--landmark_ckpt", required=True)

    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--results_csv_name",
        default="uwf_qc_results.csv",
    )
    parser.add_argument(
        "--summary_csv_name",
        default="uwf_qc_summary.csv",
    )
    parser.add_argument(
        "--qc_image_count",
        type=int,
        default=10,
        help="Save QC result images for only the first N processed rows.",
    )
    parser.add_argument(
        "--qc_image_dir_name",
        default="qc_first10",
    )

    parser.add_argument("--sample_n", type=int, default=10)
    parser.add_argument(
        "--sample_mode",
        choices=["random", "first"],
        default="random",
    )
    parser.add_argument("--random_seed", type=int, default=42)

    parser.add_argument("--img_size", type=int, default=512)

    parser.add_argument(
        "--landmark_architecture",
        default="unet",
    )
    parser.add_argument(
        "--landmark_encoder",
        default="resnet50",
    )
    parser.add_argument(
        "--roi_architecture",
        default="deeplabv3plus",
    )
    parser.add_argument(
        "--roi_encoder",
        default="resnet50",
    )
    parser.add_argument(
        "--roi_encoder_candidates",
        default="resnet50,resnet34,efficientnet-b0,timm-efficientnet-b0",
    )

    parser.add_argument("--landmark_threshold", type=float, default=0.5)
    parser.add_argument("--landmark_blob_threshold", type=float, default=0.5)
    parser.add_argument("--tta_conf_threshold", type=float, default=0.5)
    parser.add_argument("--tta_spread_threshold", type=float, default=15.0)
    parser.add_argument("--no_landmark_tta", action="store_true")

    parser.add_argument("--roi_threshold", type=float, default=0.5)
    parser.add_argument("--roi_foreground_channel", type=int, default=1)
    parser.add_argument("--roi_tta", action="store_true")

    parser.add_argument("--overwrite", action="store_true")

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    csv_path = Path(args.csv_path).expanduser().resolve()
    repo_dir = Path(args.repo_dir).expanduser().resolve()
    landmark_checkpoint = (
        Path(args.landmark_ckpt).expanduser().resolve()
    )
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    results_csv_path = output_dir / args.results_csv_name
    summary_csv_path = output_dir / args.summary_csv_name
    qc_image_dir = output_dir / args.qc_image_dir_name

    if args.qc_image_count > 0:
        qc_image_dir.mkdir(parents=True, exist_ok=True)

    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    if not repo_dir.is_dir():
        raise FileNotFoundError(
            f"Repository directory not found: {repo_dir}"
        )

    if not landmark_checkpoint.is_file():
        raise FileNotFoundError(
            f"Landmark checkpoint not found: {landmark_checkpoint}"
        )

    if (
        not args.overwrite
        and (results_csv_path.exists() or summary_csv_path.exists())
    ):
        raise FileExistsError(
            "Output CSV already exists. "
            "Use a new OUTPUT_DIR or set OVERWRITE=1 in the shell script.\n"
            f"Results: {results_csv_path}\n"
            f"Summary: {summary_csv_path}"
        )

    roi_checkpoint = find_roi_checkpoint(
        repo_dir=repo_dir,
        landmark_checkpoint=landmark_checkpoint,
        explicit_roi_checkpoint=args.roi_ckpt or None,
    )

    if not roi_checkpoint.is_file():
        raise FileNotFoundError(
            f"ROI checkpoint not found: {roi_checkpoint}"
        )

    dataframe = pd.read_csv(csv_path, low_memory=False)

    if args.image_column not in dataframe.columns:
        raise KeyError(
            f"Column '{args.image_column}' not found.\n"
            f"Available columns: {list(dataframe.columns)}"
        )

    selected = select_rows(
        dataframe=dataframe,
        image_column=args.image_column,
        sample_n=args.sample_n,
        sample_mode=args.sample_mode,
        random_seed=args.random_seed,
    )

    print(f"[CSV] total rows: {len(dataframe):,}")
    print(f"[CSV] selected rows: {len(selected):,}")
    print(f"[CSV] image column: {args.image_column}")

    if selected.empty:
        raise RuntimeError("No rows with a valid image_path value.")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"[Device] {device}")
    if device.type == "cuda":
        print(f"[GPU] {torch.cuda.get_device_name(0)}")

    # Landmark는 현재 성공한 resnet50 + 2-channel 구조를 strict load
    landmark_model, landmark_encoder_used, landmark_classes = load_smp_model(
        checkpoint_path=landmark_checkpoint,
        device=device,
        architecture=args.landmark_architecture,
        encoder_hint=args.landmark_encoder,
        encoder_candidates=args.landmark_encoder,
        fallback_classes=2,
        model_name="LANDMARK",
    )

    if landmark_classes != 2:
        print(
            f"[WARN] Landmark checkpoint output channels={landmark_classes}; "
            "channels 0 and 1 will be used as disc and fovea."
        )

    roi_model, roi_encoder_used, roi_classes = load_smp_model(
        checkpoint_path=roi_checkpoint,
        device=device,
        architecture=args.roi_architecture,
        encoder_hint=args.roi_encoder,
        encoder_candidates=args.roi_encoder_candidates,
        fallback_classes=1,
        model_name="ROI",
    )

    transform = A.Compose(
        [
            A.Resize(args.img_size, args.img_size),
            A.Normalize(
                mean=IMAGENET_DEFAULT_MEAN,
                std=IMAGENET_DEFAULT_STD,
            ),
            ToTensorV2(),
        ]
    )

    image_root = (
        Path(args.image_root).expanduser().resolve()
        if args.image_root
        else None
    )

    records: List[Dict[str, Any]] = []
    saved_qc_images: List[Path] = []

    with torch.inference_mode():
        iterator = tqdm(
            selected.to_dict(orient="records"),
            total=len(selected),
            desc="UWF QC",
        )

        for processed_index, row in enumerate(iterator):
            output_row: Dict[str, Any] = dict(row)
            output_row.update(
                {
                    "qc_status": "",
                    "qc_error": "",
                    "qc_resolved_image_path": "",
                    "image_exists": False,
                    "image_width": np.nan,
                    "image_height": np.nan,
                }
            )

            try:
                image_path = resolve_image_path(
                    row[args.image_column],
                    csv_path=csv_path,
                    image_root=image_root,
                )
                output_row["qc_resolved_image_path"] = str(image_path)
                output_row["image_exists"] = bool(image_path.is_file())

                if not image_path.is_file():
                    raise FileNotFoundError(
                        f"Image not found: {image_path}"
                    )

                bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                if bgr is None:
                    raise RuntimeError(
                        f"OpenCV could not read image: {image_path}"
                    )

                original_height, original_width = bgr.shape[:2]
                output_row["image_width"] = int(original_width)
                output_row["image_height"] = int(original_height)

                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                image_tensor = (
                    transform(image=rgb)["image"]
                    .unsqueeze(0)
                    .to(device)
                )

                save_qc_image = (
                    args.qc_image_count > 0
                    and processed_index < args.qc_image_count
                )

                roi_result = infer_roi(
                    model=roi_model,
                    image_tensor=image_tensor,
                    output_channels=roi_classes,
                    threshold=args.roi_threshold,
                    foreground_channel=args.roi_foreground_channel,
                    use_tta=args.roi_tta,
                    return_mask=save_qc_image,
                )

                landmark_result = infer_landmarks(
                    model=landmark_model,
                    image_tensor=image_tensor,
                    image_size=args.img_size,
                    original_width=original_width,
                    original_height=original_height,
                    threshold=args.landmark_threshold,
                    blob_threshold=args.landmark_blob_threshold,
                    use_tta=not args.no_landmark_tta,
                    confidence_threshold=args.tta_conf_threshold,
                    spread_threshold=args.tta_spread_threshold,
                )

                roi_mask_512 = roi_result.pop("_roi_mask_512", None)

                output_row.update(roi_result)
                output_row.update(landmark_result)
                output_row["qc_status"] = "OK"

                if save_qc_image and roi_mask_512 is not None:
                    source_name = Path(str(image_path)).name
                    safe_stem = safe_filename(
                        source_name,
                        fallback=f"row_{processed_index + 1:06d}",
                    )
                    qc_filename = (
                        f"{processed_index + 1:04d}_{safe_stem}_qc.jpg"
                    )
                    qc_path = qc_image_dir / qc_filename

                    qc_visual = make_qc_visual(
                        original_bgr=bgr,
                        roi_mask_512=roi_mask_512,
                        roi_result=roi_result,
                        landmark_result=landmark_result,
                        display_name=source_name,
                    )

                    if not cv2.imwrite(str(qc_path), qc_visual):
                        raise RuntimeError(
                            f"Failed to write QC image: {qc_path}"
                        )

                    saved_qc_images.append(qc_path)
                    output_row["qc_result_image"] = str(qc_path)
                else:
                    output_row["qc_result_image"] = ""

            except Exception as exc:
                output_row["qc_status"] = "ERROR"
                output_row["qc_error"] = (
                    f"{type(exc).__name__}: {exc}"
                )

                # 실패 행에도 열이 존재하도록 기본값 추가
                defaults = {
                    "roi_detected": False,
                    "roi_area_pixels_512": np.nan,
                    "roi_area_ratio": np.nan,
                    "roi_mean_confidence": np.nan,
                    "roi_max_confidence": np.nan,
                    "roi_foreground_channel": np.nan,
                    "disc_detected": False,
                    "fovea_detected": False,
                    "both_disc_fovea_detected": False,
                    "neither_disc_fovea_detected": False,
                    "disc_x": np.nan,
                    "disc_y": np.nan,
                    "disc_center": "",
                    "fovea_x": np.nan,
                    "fovea_y": np.nan,
                    "fovea_center": "",
                    "disc_confidence": np.nan,
                    "fovea_confidence": np.nan,
                    "disc_tta_mean_confidence": np.nan,
                    "fovea_tta_mean_confidence": np.nan,
                    "disc_tta_mean_distance_512": np.nan,
                    "fovea_tta_mean_distance_512": np.nan,
                    "disc_tta_detection_count": 0,
                    "fovea_tta_detection_count": 0,
                    "disc_status": "Error",
                    "fovea_status": "Error",
                    "qc_result_image": "",
                }
                for key, value in defaults.items():
                    output_row.setdefault(key, value)

            records.append(output_row)

    if saved_qc_images:
        contact_sheet_path = qc_image_dir / "qc_first10_contact_sheet.jpg"
        save_contact_sheet(
            image_paths=saved_qc_images,
            output_path=contact_sheet_path,
            columns=2,
        )
        print(f"[QC images] saved: {len(saved_qc_images)}")
        print(f"[QC images] directory: {qc_image_dir}")
        print(f"[QC images] contact sheet: {contact_sheet_path}")

    result_dataframe = pd.DataFrame(records)
    result_dataframe.to_csv(
        results_csv_path,
        index=False,
        encoding="utf-8-sig",
    )

    summary_dataframe = build_summary(
        result_dataframe=result_dataframe,
        csv_path=csv_path,
        output_path=results_csv_path,
        roi_checkpoint=roi_checkpoint,
        landmark_checkpoint=landmark_checkpoint,
        roi_encoder=roi_encoder_used,
        landmark_encoder=landmark_encoder_used,
    )
    summary_dataframe.to_csv(
        summary_csv_path,
        index=False,
        encoding="utf-8-sig",
    )

    ok = result_dataframe["qc_status"].eq("OK")
    denominator = int(ok.sum())

    both = int(
        result_dataframe.loc[
            ok, "both_disc_fovea_detected"
        ].fillna(False).sum()
    )
    disc_not = int(
        (
            ~result_dataframe.loc[
                ok, "disc_detected"
            ].fillna(False)
        ).sum()
    )
    fovea_not = int(
        (
            ~result_dataframe.loc[
                ok, "fovea_detected"
            ].fillna(False)
        ).sum()
    )

    print()
    print("=" * 68)
    print(f"[DONE] results: {results_csv_path}")
    print(f"[DONE] summary: {summary_csv_path}")
    print(f"Successfully processed: {denominator}/{len(result_dataframe)}")

    if denominator > 0:
        print(
            f"Both disc + fovea detected: "
            f"{both}/{denominator} ({both / denominator:.2%})"
        )
        print(
            f"Disc not detected: "
            f"{disc_not}/{denominator} ({disc_not / denominator:.2%})"
        )
        print(
            f"Fovea not detected: "
            f"{fovea_not}/{denominator} ({fovea_not / denominator:.2%})"
        )
    print("=" * 68)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
