#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def pct(n, d):
    return 100.0 * n / d if d > 0 else 0.0


def load_and_concat(csv_list):
    dfs = []
    for idx,p in enumerate(csv_list):
        df = pd.read_csv(p)
        df["__source_csv_idx__"] = idx   
        df["__source_csv_path__"] = str(Path(p).resolve())
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def mean_std(x):
    x = x.dropna()
    if len(x) == 0:
        return np.nan, np.nan, 0
    return float(x.mean()), float(x.std(ddof=1)), int(len(x))

def zone1_visibility(df_part, z1_col, eps):
    valid = df_part[z1_col].notna()
    total = len(df_part)
    n_full = int((np.abs(df_part.loc[valid, z1_col] - 1.0) <= eps).sum())
    return n_full, total, pct(n_full, total)

def list_zone1_failures(df_part, z1_col, eps, name_col="image_fname", stem_only=True):
    """
    Return list of identifiers for rows where Zone1 ratio != 1 (within eps).
    """
    valid = df_part[z1_col].notna()
    fail = valid & (np.abs(df_part[z1_col] - 1.0) > eps)

    if name_col not in df_part.columns:
        # fallback: return index
        return df_part.index[fail].astype(str).tolist()

    names = df_part.loc[fail, name_col].astype(str).tolist()
    if stem_only:
        names = [Path(n).stem for n in names]
    return names


def zone2_valid_ratio(df_part, z1_col, z2_cols, eps):
    total = len(df_part)
    valid = df_part[z2_cols].notna().all(axis=1)

    z2_all_one = (
        np.abs(df_part.loc[valid, z2_cols] - 1.0) <= eps
    ).all(axis=1)

    n_full = int(z2_all_one.sum())
    return n_full, total, pct(n_full, total)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csvs", nargs="+", required=True,
                    help="List of zone_stats.csv files (3 or more).")
    ap.add_argument("--eps", type=float, default=1e-6,
                    help="Tolerance for checking ratio==1.")
    args = ap.parse_args()

    # -------------------------------------------------
    # Load & merge
    # -------------------------------------------------
    df = load_and_concat(args.csvs)
    # -----------------------------------------
    # Split Internal / External
    # -----------------------------------------
    csv_names = [Path(p).name for p in args.csvs]

    if len(csv_names) < 3:
        raise RuntimeError(
            "Expected CSV order: [Internal, External1, External2]"
        )

    internal_csv = csv_names[0]
    external_csvs = set(csv_names[1:])

    df_internal = df[df["__source_csv_idx__"] == 0].copy()
    df_external = df[df["__source_csv_idx__"].isin([1, 2])].copy()

    print(f"[DEBUG] Internal rows  = {len(df_internal)}")
    print(f"[DEBUG] External rows  = {len(df_external)}")

    # ratio columns
    z1 = "Zone1_ratio_A_div_B"
    z2_cols = [
        "Zone2T_ratio_A_div_B",
        "Zone2S_ratio_A_div_B",
        "Zone2N_ratio_A_div_B",
        "Zone2I_ratio_A_div_B",
    ]

    # ensure numeric
    for c in [z1] + z2_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    valid = df[z1].notna()
    total = int(valid.sum())
    if total == 0:
        raise RuntimeError("No valid rows found.")

    # -------------------------------------------------
    # Zone 1 Visibility
    # -------------------------------------------------
    n1_i, tot1_i, z1_i = zone1_visibility(df_internal, z1, args.eps)
    n1_e, tot1_e, z1_e = zone1_visibility(df_external, z1, args.eps)

    fail_internal = list_zone1_failures(df_internal, z1, args.eps, name_col="image_fname", stem_only=False)
    fail_external = list_zone1_failures(df_external, z1, args.eps, name_col="image_fname", stem_only=False)

    print()
    print(f"[Zone1 NOT FULL] Internal: {len(fail_internal)} cases")
    for n in fail_internal:
        print("  -", n)

    print()
    print(f"[Zone1 NOT FULL] External: {len(fail_external)} cases")
    for n in fail_external:
        print("  -", n)

    # -------------------------------------------------
    # Zone 2 Valid Ratio (all four == 1)
    # -------------------------------------------------
    n2_i, tot2_i, z2_i = zone2_valid_ratio(df_internal, z1, z2_cols, args.eps)
    n2_e, tot2_e, z2_e = zone2_valid_ratio(df_external, z1, z2_cols, args.eps)


    # -------------------------------------------------
    # Zone 2 mean / std
    # -------------------------------------------------
    z2_stats = {}
    for c in z2_cols:
        m, s, n = mean_std(df.loc[valid, c])
        z2_stats[c] = (m, s, n)

    # -------------------------------------------------
    # Zone 3 Artifact Ratio
    # -------------------------------------------------
    z3A_cols = [
        "Zone3T_A_gt_in_zone", "Zone3S_A_gt_in_zone",
        "Zone3N_A_gt_in_zone", "Zone3I_A_gt_in_zone",
    ]
    z3B_cols = [
        "Zone3T_B_zone_area", "Zone3S_B_zone_area",
        "Zone3N_B_zone_area", "Zone3I_B_zone_area",
    ]

    for c in z3A_cols + z3B_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    z3A = df[z3A_cols].sum(axis=1)
    z3B = df[z3B_cols].sum(axis=1)

    z3_roi_ratio = z3A / z3B
    z3_artifact_ratio = 1.0 - z3_roi_ratio

    z3_roi_ratio = z3_roi_ratio[valid & z3B.notna() & (z3B > 0)]
    z3_artifact_ratio = z3_artifact_ratio[valid & z3B.notna() & (z3B > 0)]

    z3_roi_mean, z3_roi_std, _ = mean_std(z3_roi_ratio)
    z3_art_mean, z3_art_std, _ = mean_std(z3_artifact_ratio)

    # -------------------------------------------------
    # Print summary
    # -------------------------------------------------
    print("==============================================")
    print(" Zone Statistics (Merged CSVs)")
    print("==============================================")
    print(f"Total images: {total}")
    print()
    print("==============================================")
    print(" Zone Visibility / Valid Ratio")
    print("==============================================")

    print("[Internal]")
    print(f"  Zone 1 Visibility: {n1_i}/{tot1_i} = {z1_i:.2f}%")
    print(f"  Zone 2 Valid Ratio: {n2_i}/{tot2_i} = {z2_i:.2f}%")
    print()

    print("[External]")
    print(f"  Zone 1 Visibility: {n1_e}/{tot1_e} = {z1_e:.2f}%")
    print(f"  Zone 2 Valid Ratio: {n2_e}/{tot2_e} = {z2_e:.2f}%")

    print("==============================================")


    print("Zone 2 valid area ratio (mean ± std):")
    for c, (m, s, n) in z2_stats.items():
        name = c.replace("_ratio_A_div_B", "")
        print(f"  {name}: {m:.4f} ± {s:.4f}  (N={n})")

    print()
    print("Zone 3 Artifact Analysis:")
    print(f"  Zone 3 ROI ratio mean ± std:      {z3_roi_mean:.4f} ± {z3_roi_std:.4f}")
    print(f"  Zone 3 Artifact ratio mean ± std: {z3_art_mean:.4f} ± {z3_art_std:.4f}")
    print("==============================================")


if __name__ == "__main__":
    main()
