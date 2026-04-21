import argparse
import random
import shutil
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--image_dir", type=str)
    parser.add_argument("--save_dir", type=str)
    return parser.parse_args()


def main():
    args = parse_args()

    image_dir = Path(args.image_dir)
    save_dir = Path(args.save_dir)

    save_dir.mkdir(parents=True, exist_ok=True)

    exts = [".jpg"]
    image_paths = sorted([p for p in image_dir.rglob("*") if p.suffix.lower() in exts])

    if not image_paths:
        print(f"[ERROR] no images contained in {image_dir}")
        return

    print(f"[INFO] total images = {len(image_paths)}")

    # 샘플링
    sampled = random.sample(image_paths, 25)
    print(f"[INFO] sampled images = {len(sampled)}")

    # 복사
    for src in sampled:
        dst = save_dir / src.name 
        shutil.copy(src, dst)
        print(f"[COPY] {src}  ->  {dst}")

    print("[DONE] sampling & copy complete.")


if __name__ == "__main__":
    main()
