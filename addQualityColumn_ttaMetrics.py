import pandas as pd
from pathlib import Path
import argparse


def add_quality_column(
    csv_path,
    poor_quality_dir,
    output_csv= None,
):
    df = pd.read_csv(csv_path)

    if "stem" not in df.columns:
        raise ValueError("CSV 파일에 'stem' 열이 존재하지 않습니다.")

    def check_quality(stem):
        img_path = poor_quality_dir / f"{stem}.jpg"
        return 0 if img_path.exists() else 1

    df["Quality"] = df["stem"].apply(check_quality)

    if output_csv is None:
        output_csv = csv_path

    df.to_csv(output_csv, index=False)
    print(f"Saved CSV with Quality column → {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="입력 CSV 파일 경로")
    parser.add_argument(
        "--poor_quality_dir",
        default="/mnt/richul_FM/UWF_seg_det/datasets/Det/OUWFD/poor_quality_images",
        help="poor quality 이미지 폴더 경로",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="출력 CSV 경로 (미지정 시 입력 CSV 덮어씀)",
    )

    args = parser.parse_args()

    add_quality_column(
        csv_path=Path(args.csv),
        poor_quality_dir=Path(args.poor_quality_dir),
        output_csv=Path(args.out) if args.out else None,
    )
