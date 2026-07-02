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
