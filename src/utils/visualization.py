import numpy as np
import matplotlib.pyplot as plt


def visualize_random_volume(volume, slice_idx=None):
    assert volume.ndim == 3, "Volume must be 3D (D, H, W)"

    D, H, W = volume.shape

    if slice_idx is None:
        slice_idx = D // 2  

    slice_img = volume[slice_idx]

    plt.figure(figsize=(6, 6))
    plt.imshow(slice_img, cmap="gray")
    plt.title(f"Random Volume Slice #{slice_idx}")
    plt.axis("off")
    plt.show()


def visualize_volume_with_sparse_mask(volume, sparse_mask, slice_idx=None):
    assert volume.ndim == 3, "Volume must be 3D (D, H, W)"
    assert sparse_mask.ndim == 3, "Mask must be 3D (D, H, W)"
    assert volume.shape == sparse_mask.shape, "Volume and mask shape mismatch"

    D, H, W = volume.shape

    if slice_idx is None:
        slice_idx = D // 2

    img = volume[slice_idx]
    mask = sparse_mask[slice_idx]

    plt.figure(figsize=(6, 6))
    plt.imshow(img, cmap="gray")
    plt.imshow(mask, cmap="jet", alpha=0.4)  
    plt.title(f"Volume + Sparse Mask Slice #{slice_idx}")
    plt.axis("off")
    plt.show()
