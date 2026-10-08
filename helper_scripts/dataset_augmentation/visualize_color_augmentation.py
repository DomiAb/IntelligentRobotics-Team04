"""
Just a easy script to check out the colour agemntations that will be done on the fly during training.
Paramters are hard coded as this is just intended for a quick check.
"""

import h5py
import matplotlib.pyplot as plt
import torch
import torchvision.transforms.functional as F

# 1. Extract a single test frame from your robomimic dataset
dataset_path = "dataset/square_splits/square_train.hdf5"
with h5py.File(dataset_path, "r") as f:
    # Read frame 0 of the first demonstration, convert to channel-first tensor [C, H, W]
    raw_img = f["data/demo_0/obs/agentview_image"][0]
    img_tensor = torch.from_numpy(raw_img).permute(2, 0, 1).float() / 255.0

# 2. Compute the exact boundary transformations
transforms_dict = {
    "Original (Neutral)": img_tensor,
    "Brightness Min (0.8)": F.adjust_brightness(img_tensor, 0.6),
    "Brightness Max (1.2)": F.adjust_brightness(img_tensor, 1.4),
    "Contrast Min (0.8)": F.adjust_contrast(img_tensor, 0.6),
    "Contrast Max (1.2)": F.adjust_contrast(img_tensor, 1.4),
    "Saturation Min (0.8)": F.adjust_saturation(img_tensor, 0.6),
    "Saturation Max (1.2)": F.adjust_saturation(img_tensor, 1.4),
    "Hue Min (-0.02)": F.adjust_hue(img_tensor, -0.04),
    "Hue Max (+0.02)": F.adjust_hue(img_tensor, 0.04),
}

# 3. Plot the diagnostic comparison matrix
fig, axes = plt.subplots(3, 3, figsize=(10, 10))
axes = axes.flatten()

for i, (title, transformed_tensor) in enumerate(transforms_dict.items()):
    # Convert back to channel-last [H, W, C] for matplotlib display
    display_img = transformed_tensor.permute(1, 2, 0).numpy()
    axes[i].imshow(display_img)
    axes[i].set_title(title, fontsize=10)
    axes[i].axis("off")

plt.tight_layout()
plt.savefig("color_jitter_bounds_check.png", dpi=300)
print("Saved comparison grid to color_jitter_bounds_check.png in project root")