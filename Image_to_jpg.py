#!/usr/bin/env python3
import os
from pathlib import Path
from PIL import Image
import argparse

def resave_as_jpg(input_dir):
    input_dir = Path(input_dir)

    if not input_dir.exists():
        print(f"[ERROR] Folder not found: {input_dir}")
        return

    IMAGE_EXTS = {".png", ".jpg"}
    image_paths = sorted([p for p in input_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS])

    if not image_paths:
        print("[INFO] No image files found.")
        return

    print(f"[INFO] Found {len(image_paths)} image files.")

    for img_path in image_paths:
        try:
            img = Image.open(img_path)
            img = img.convert("RGB")  # 모든 이미지 RGB로 통일 (저장 안정성 ↑)

            save_path = img_path.with_suffix(".jpg")

            img.save(save_path, quality=95)
            print(f"[OK] Saved JPG: {save_path}")

            try:
                img_path.unlink()
                print(f"[DEL] Deleted original: {img_path}")
            except Exception as e:
                print(f"[ERROR] Failed to delete {img_path}: {e}")


        except Exception as e:
            print(f"[ERROR] Failed to process {img_path}: {e}")

    print("[DONE] All images processed.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, required=True,
                        help="Path to the folder containing images to be re-saved as JPG.")
    args = parser.parse_args()

    resave_as_jpg(args.folder)


if __name__ == "__main__":
    main()
