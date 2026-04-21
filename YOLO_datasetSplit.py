
import argparse
import random
import re
import shutil
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

IMG_EXTS = {".jpg", ".png"}

# OTANONY00014997_20210107_....
PID_DATE_REGEX = re.compile(r"^OTANONY(?P<pid>\d{8})_(?P<date>\d{8})_")

def safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def list_images(d: Path) -> List[Path]:
    if not d.exists():
        return []
    return sorted(
        p for p in d.iterdir()
        if p.is_file() and p.suffix.lower() in IMG_EXTS
    )

def extract_patient_and_date(filename: str) -> Tuple[str, str]:
    m = PID_DATE_REGEX.match(filename)
    if not m:
        raise ValueError(f"Cannot parse PatientID/Exam_Date from filename: {filename}")
    return m.group("pid"), m.group("date")

def corresponding_label_path(img_path: Path, labels_dir: Path) -> Path:
    return labels_dir / f"{img_path.stem}.txt"

def copy_or_move(src: Path, dst: Path, do_move: bool) -> None:
    safe_mkdir(dst.parent)
    if do_move:
        shutil.move(str(src), str(dst))
    else:
        shutil.copy2(str(src), str(dst))

def write_split_csv(csv_path: Path, rows: List[Dict[str, str]]) -> None:
    safe_mkdir(csv_path.parent)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["img_fname", "mask_fname", "Exam_Date", "PatientID"])
        w.writeheader()
        w.writerows(rows)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root", type=str, default="/mnt/richul_FM/YOLO/datasets",
        help="datasets root path (contains images/ and labels/)"
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--move", action="store_true",
        help="move files instead of copy (WARNING: modifies original folders)"
    )

    args = ap.parse_args()

    root = Path(args.root)
    input_images_root = root / "images"
    input_labels_root = root / "labels"

    save_root = root / "new_split_"
    images_root = save_root / "images"
    labels_root = save_root / "labels"
    csv_root = save_root / "csv"

    # Flat input structure
    src_pairs = [
        (input_images_root, input_labels_root),
    ]

    groups: Dict[str, List[Tuple[Path, Path, str]]] = defaultdict(list)  # pid -> [(img, lbl, exam_date), ...]

    total_found_imgs = 0
    total_used_imgs = 0
    parse_fail = 0
    missing_label_cnt = 0

    for img_dir, lbl_dir in src_pairs:
        for img_path in list_images(img_dir):
            total_found_imgs += 1

            try:
                pid, exam_date = extract_patient_and_date(img_path.name)
            except ValueError:
                parse_fail += 1
                continue

            lbl_path = corresponding_label_path(img_path, lbl_dir)
            if not lbl_path.exists():
                missing_label_cnt += 1
                continue

            groups[pid].append((img_path, lbl_path, exam_date))
            total_used_imgs += 1

    if total_used_imgs == 0:
        raise RuntimeError("No labeled images available after filtering.")

    patient_ids = list(groups.keys())
    random.seed(args.seed)
    random.shuffle(patient_ids)

    total_imgs = sum(len(groups[pid]) for pid in patient_ids)

    TARGET_VALID = 150
    TARGET_TEST  = 150

    target_valid = min(TARGET_VALID, total_imgs)
    target_test  = min(TARGET_TEST, total_imgs - target_valid)
    target_train = total_imgs - target_valid - target_test

    split_pids = {"train": [], "valid": [], "test": []}
    split_counts = {"train": 0, "valid": 0, "test": 0}
    targets = {"train": target_train, "valid": target_valid, "test": target_test}

    def deficit(split: str) -> int:
        return targets[split] - split_counts[split]

    for pid in patient_ids:
        gsz = len(groups[pid])

        candidates = ["train", "valid", "test"]
        candidates.sort(
            key=lambda s: (deficit(s),
                           1 if s == "train" else 0 if s == "valid" else -1),
            reverse=True
        )
        chosen = candidates[0]
        split_pids[chosen].append(pid)
        split_counts[chosen] += gsz

    for split in ["train", "valid", "test"]:
        safe_mkdir(images_root / split)
        safe_mkdir(labels_root / split)

    # Collect CSV rows per split
    csv_rows = {"train": [], "valid": [], "test": []}

    for split in ["train", "valid", "test"]:
        for pid in split_pids[split]:
            for img_src, lbl_src, exam_date in groups[pid]:
                img_dst = images_root / split / img_src.name
                lbl_dst = labels_root / split / f"{img_src.stem}.txt"

                copy_or_move(img_src, img_dst, args.move)
                copy_or_move(lbl_src, lbl_dst, args.move)

                img_fname = img_src.name
                csv_rows[split].append({
                    "img_fname": img_fname,
                    "mask_fname": img_fname,  # requested: same value as img_fname
                    "Exam_Date": exam_date,
                    "PatientID": pid,
                })

    # Write {train|valid|test}.csv into save_root
    for split in ["train", "valid", "test"]:
        write_split_csv(csv_root / f"{split}.csv", csv_rows[split])

    print("Done.")
    print(f"Found images: {total_found_imgs}")
    print(f"Used images (labeled only): {total_used_imgs}")
    print(f"Patients: {len(patient_ids)}")
    print(f"Filename parse failures skipped: {parse_fail}")
    print(f"Images skipped due to missing label: {missing_label_cnt}")
    print("")
    print("Targets (by images):")
    print(f"  train={target_train}, valid={target_valid}, test={target_test} (total={total_imgs})")
    print("Actual (by images):")
    print(f"  train={split_counts['train']}, valid={split_counts['valid']}, test={split_counts['test']}")
    print("Actual (by patients):")
    print(f"  train={len(split_pids['train'])}, valid={len(split_pids['valid'])}, test={len(split_pids['test'])}")
    print(f"mode: {'MOVE' if args.move else 'COPY'}")
    print(f"CSV saved to: {save_root}/{{train,valid,test}}.csv")

if __name__ == "__main__":
    main()
