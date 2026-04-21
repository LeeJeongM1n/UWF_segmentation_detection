import numpy as np

paths = [
    "/mnt/richul_FM/UWF_seg_det/datasets/Det/inference_output/2class/sigma20/cropCircle/npy/angles_abs_internal.npy",
    "/mnt/richul_FM/UWF_seg_det/datasets/Det/inference_output/2class/sigma20/cropCircle/npy/angles_abs_OUWFD.npy",
    "/mnt/richul_FM/UWF_seg_det/datasets/Det/inference_output/2class/sigma20/cropCircle/npy/angles_abs_MSHF.npy",
]

all_angles = []

for p in paths:
    a = np.load(p)
    all_angles.append(a)

all_angles = np.concatenate(all_angles, axis=0)

mean_abs = all_angles.mean()
std_abs  = all_angles.std(ddof=1)   # sample std
min_abs  = all_angles.min()
max_abs  = all_angles.max()

print(
    f"[COMBINED] N={len(all_angles)} | "
    f"mean(|angle|)={mean_abs:.2f}° ± {std_abs:.2f}° "
    f"(range: {min_abs:.2f}° to {max_abs:.2f}°)"
)
