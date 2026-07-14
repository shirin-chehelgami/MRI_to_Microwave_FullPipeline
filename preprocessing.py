"""
preprocessing.py — image preprocessing for this package.

POLICY (shared by all packages):
  * z-score normalization is ALWAYS applied (Duke seg-model requirement):
        quantile-clip[0.1%,99.9%] -> min-max[0,1] -> z-score(mean0/std1)
  * a per-package DENOISE step is applied in the [0,1] domain BEFORE z-scoring.

This file is the ONLY difference between the three packages.
"""
import numpy as np
from scipy.ndimage import median_filter

import SimpleITK as sitk 
from scipy import ndimage as _ndi


def normalize_image(arr, min_cutoff=0.001, max_cutoff=0.001):
    a = arr.astype(np.float32)
    lo = float(np.quantile(a, min_cutoff)); hi = float(np.quantile(a, 1.0 - max_cutoff))
    a = np.clip(a, lo, hi)
    return (a - lo) / max(hi - lo, 1e-8)

def zscore_image(arr):
    a = arr.astype(np.float32)
    return (a - float(a.mean())) / max(float(a.std()), 1e-8)

# cross/plus footprint: centre + 6 face neighbours (edge-preserving)
_PLUS = np.zeros((3,3,3), bool); _PLUS[1,1,1] = True
for _a in range(3):
    for _d in (0,2):
        _i = [1,1,1]; _i[_a] = _d; _PLUS[tuple(_i)] = True

def switching_median(x, k=3.0):
    """Decision-based median: replace a voxel by its local (cross) median ONLY if it
    deviates by more than k*sigma — removes impulse specks, leaves the rest untouched."""
    med = median_filter(x, footprint=_PLUS)
    r = x - med
    sigma = 1.4826 * np.median(np.abs(r - np.median(r)))
    return np.where(np.abs(x - med) > k*sigma, med, x)



def _denoise(v01):
    """switching-median package: edge-preserving impulse removal (k=3)."""
    return switching_median(v01, k=3.0)

def preprocess(vol):
    return zscore_image(_denoise(normalize_image(vol)))


# ---------- N4 helpers (called later, on the DIELECTRIC image, not here) ----------
def _to_positive(vol):
    v = vol.astype("float32")
    off = (-float(v.min()) + 0.1) if v.min() <= 0 else 0.0   # z-scored imgs have negatives
    return v + off, off

def _n4_divide(v_pos, mask_arr, levels, shrink):
    im = sitk.Cast(sitk.GetImageFromArray(v_pos), sitk.sitkFloat32)
    m  = sitk.Cast(sitk.GetImageFromArray(mask_arr.astype("uint8")), sitk.sitkUInt8)
    n4 = sitk.N4BiasFieldCorrectionImageFilter()
    n4.SetMaximumNumberOfIterations([50] * levels)
    n4.Execute(sitk.Shrink(im, [shrink]*3), sitk.Shrink(m, [shrink]*3))
    bias = sitk.Exp(n4.GetLogBiasFieldAsImage(im))
    return sitk.GetArrayFromImage(im / bias)

def apply_n4_whole(vol, tissue_mask, levels=4, shrink=4):
    """N4 over the whole breast (masked to tissue)."""
    v, off = _to_positive(vol)
    corr = _n4_divide(v, tissue_mask, levels, shrink)
    return (corr - off).astype(vol.dtype)

def apply_n4_edge(vol, label, spacing_zyx, feather_mm=30.0, levels=4, shrink=4,
                  FAT=1, MUSCLE=4, SKIN=5):
    """Skin-edge N4: smooth field fit on FAT, applied feathered from the SKIN inward
    (feather_mm), chest-wall (MUSCLE) excluded. Deep interior & chest wall untouched."""
    def _ss(x): x = np.clip(x, 0, 1); return x*x*(3 - 2*x)
    v, off = _to_positive(vol)
    corr = _n4_divide(v, (label == FAT), levels, shrink)
    dist_skin = _ndi.distance_transform_edt(label != SKIN, sampling=spacing_zyx)
    w = _ss((feather_mm - dist_skin) / feather_mm) * (label > 0)
    w[label == MUSCLE] = 0.0
    return ((1.0 - w) * v + w * corr - off).astype(vol.dtype)