

import h5py
import matplotlib.pyplot as plt
import numpy as np

hdf5_path = "/home/smlr_g2/smlr_g2_WS25/deoxys_control/deoxys/g2_repo/data/dataset/square_d0_dr0_two_views.hdf5"   # 改成你的路径
demo_name = "demo_141_dr_1"       # 改成你想看的 demo

with h5py.File(hdf5_path, "r") as f:
    imgs = f[f"data/{demo_name}/obs/robot0_eye_in_hand_image"][:]

print("imgs shape:", imgs.shape)   # (T, 96, 96, 3)

num_show = 6
indices = np.linspace(0, len(imgs) - 1, num_show, dtype=int)

plt.figure(figsize=(12, 8))
for i, idx in enumerate(indices):
    plt.subplot(2, 3, i + 1)
    plt.imshow(imgs[idx])
    plt.title(f"{demo_name}\nframe {idx}")
    plt.axis("off")

plt.tight_layout()
plt.show()