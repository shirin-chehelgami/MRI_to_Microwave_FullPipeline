import gzip, struct
import numpy as np
from scipy.ndimage import label as nd_label
from scipy import ndimage
from scipy.ndimage import binary_dilation

MUSCLE_REAL = 50.0
MUSCLE_IMAG = 5.0

def read_nii_gz(path):
    with gzip.open(str(path), 'rb') as f:
        raw = f.read()
    vox_offset = int(struct.unpack_from('<f', raw, 108)[0])
    dims = struct.unpack_from('<8h', raw, 40)
    datatype = struct.unpack_from('<h', raw, 70)[0]
    pixdim = struct.unpack_from('<8f', raw, 76)
    shape = (dims[3], dims[2], dims[1])  # (Z, Y, X)
    dtype_map = {2: np.uint8, 4: np.int16, 8: np.int32, 16: np.float32, 64: np.float64}
    data = np.frombuffer(raw[vox_offset:], dtype=dtype_map.get(datatype, np.float32)).reshape(shape)
    return data, pixdim

def pad_volume_3d(data, pad=20):
    Z, Y, X = data.shape
    padded = np.zeros((Z + 2*pad, Y + 2*pad, X + 2*pad), dtype=data.dtype)
    padded[pad:pad+Z, pad:pad+Y, pad:pad+X] = data
    return padded

def find_split_matlab_exact(mask):
    """
    Exact Python equivalent of MATLAB logic.
    D3 = A→P = X axis in your data (sagittal slices)
    D1 = R→L = Y axis in your data (the split axis)
    """
    D1, D2, D3 = mask.shape  # z, y, x
    # Step 1: scan along X to find two-component slices
    first_two = None
    last_two = None
    for s in range(D3):
        bw = mask[:, :, s]          # 2D slice in (z, y) plane
        _, n = nd_label(bw)
        if n == 2:
            if first_two is None:
                first_two = s
            last_two = s
        elif n == 1 and first_two is not None:
            break
    print(f'First two-component X slice: {first_two}')
    print(f'Last  two-component X slice: {last_two}')

    # Step 2: middle slice
    mid_slice = (first_two + last_two) // 2
    print(f'Middle X slice             : {mid_slice}')

    # Step 3: connected components on middle slice
    bw_mid = mask[:, :, mid_slice]  # shape (z, y)
    labeled_mid, n = nd_label(bw_mid)
    if n < 2:
        raise ValueError(f'Middle slice has only {n} component(s)')

    # Keep 2 largest
    sizes = ndimage.sum(bw_mid, labeled_mid, range(1, n+1))
    top2 = np.argsort(sizes)[-2:] + 1

    # Step 4: centroid of each component  (returns (z, y))
    centroids = ndimage.center_of_mass(bw_mid, labeled_mid, top2)
    c1_y = centroids[0][1]
    c2_y = centroids[1][1]

    # D1 (Y) increases R→L: smaller Y = right breast
    if c1_y < c2_y:
        right_centroid_y, left_centroid_y = c1_y, c2_y
    else:
        right_centroid_y, left_centroid_y = c2_y, c1_y
    print(f'Right centroid Y           : {right_centroid_y:.1f}')
    print(f'Left  centroid Y           : {left_centroid_y:.1f}')

    # Step 5: split at midpoint
    split_y = int(round((right_centroid_y + left_centroid_y) / 2))
    print(f'Split Y                    : {split_y}')
    return split_y, right_centroid_y, left_centroid_y, mid_slice

def find_inner_edges(mask, right_cy, left_cy, margin_frac=0.05):
    """
    Find two global split lines from the Y tissue profile:
      b1_end   = inner edge of breast 1 (where breast 1 tissue ends)
      b2_start = inner edge of breast 2 (where breast 2 tissue starts)

    Breast 1 keeps everything with Y < b2_start, breast 2 keeps everything
    with Y > b1_end, so each breast retains the full gap/valley between them.
    """
    Y = mask.shape[1]
    lo = int(round(min(right_cy, left_cy)))
    hi = int(round(max(right_cy, left_cy)))

    prof = mask.sum(axis=(0, 2)).astype(float)   # tissue summed over Z and X
    seg = prof[lo:hi+1]
    vi = int(np.argmin(seg))                      # valley index within [lo, hi]
    valley = lo + vi
    thresh = seg[vi] + max(1.0, margin_frac * seg.max())

    # breast 1 ends: scan from valley toward breast 1 (decreasing Y)
    b1_end = valley
    for i in range(vi, -1, -1):
        if seg[i] > thresh:
            b1_end = lo + i
            break
    # breast 2 starts: scan from valley toward breast 2 (increasing Y)
    b2_start = valley
    for i in range(vi, len(seg)):
        if seg[i] > thresh:
            b2_start = lo + i
            break

    print(f'Valley Y                   : {valley}')
    print(f'Breast 1 inner edge (end)  : {b1_end}')
    print(f'Breast 2 inner edge (start): {b2_start}')
    return b1_end, b2_start, valley

def add_muscle(breast_mask_single, max_x_global, side, min_row_voxels=5):
    muscle = np.zeros_like(breast_mask_single)
    Z, Y, X = breast_mask_single.shape
    for z in range(Z):
        sl = breast_mask_single[z, :, :]
        if not sl.any():
            continue
        row_counts = sl.sum(axis=1)
        real_rows = np.where(row_counts >= min_row_voxels)[0]
        if len(real_rows) == 0:
            continue
        real_first_y = real_rows.min()
        real_last_y = real_rows.max()

        deep_x_per_y = np.full(Y, -1, dtype=int)
        for y in range(Y):
            xs = np.where(sl[y, :])[0]
            if len(xs) > 0:
                deep_x_per_y[y] = xs.max()

        for y in range(real_first_y, real_last_y + 1):
            dx = deep_x_per_y[y]
            if dx >= 0 and dx + 1 <= max_x_global:
                muscle[z, y, dx + 1 : max_x_global + 1] = True
    muscle &= ~breast_mask_single
    return muscle


def add_muscle_contour(breast_mask_single, thickness_vox, min_row_voxels=5):
    """Contoured muscle: a constant-thickness layer following the breast's POSTERIOR
    surface (the deepest breast voxel per row), extruded back by `thickness_vox`.
    Unlike add_muscle (flat back wall), this hugs the breast's actual back contour."""
    muscle = np.zeros_like(breast_mask_single)
    Z, Y, X = breast_mask_single.shape
    for z in range(Z):
        sl = breast_mask_single[z, :, :]
        if not sl.any():
            continue
        row_counts = sl.sum(axis=1)
        real_rows = np.where(row_counts >= min_row_voxels)[0]
        if len(real_rows) == 0:
            continue
        for y in range(real_rows.min(), real_rows.max() + 1):
            xs = np.where(sl[y, :])[0]
            if len(xs) == 0:
                continue
            dx = xs.max()                      # posterior surface of breast at this (z,y)
            x_end = min(dx + thickness_vox, X - 1)
            muscle[z, y, dx + 1 : x_end + 1] = True   # exactly thickness_vox behind the surface
    muscle &= ~breast_mask_single
    return muscle

def extend_muscle_padded(muscle, thickness=25):
    extended = muscle.copy()
    Z, Y, X = muscle.shape
    for z in range(Z):
        for y in range(Y):
            xs = np.where(muscle[z, y, :])[0]
            if len(xs) == 0:
                continue
            curr = xs.max()
            new_max = min(curr + thickness, X-1)
            if new_max > curr:
                extended[z, y, curr+1:new_max+1] = True
    return extended

def extend_muscle_yz(muscle, thickness_y=15, thickness_z=8):
    struct = np.ones((2*thickness_z+1, 2*thickness_y+1, 1), dtype=bool)
    return binary_dilation(muscle, structure=struct)

def crop_to_content(*arrs, pad=5):
    nz = np.zeros(arrs[0].shape, dtype=bool)
    for a in arrs:
        nz |= (a != 0)
    cz, cy, cx = np.where(nz)
    z0, z1 = max(0, cz.min()-pad), min(arrs[0].shape[0], cz.max()+pad+1)
    y0, y1 = max(0, cy.min()-pad), min(arrs[0].shape[1], cy.max()+pad+1)
    x0, x1 = max(0, cx.min()-pad), min(arrs[0].shape[2], cx.max()+pad+1)
    slices = (slice(z0,z1), slice(y0,y1), slice(x0,x1))
    return [a[slices] for a in arrs], slices



def crop_to_content_asym(*arrs, pad=5, back_pad_x=8):
    """Like crop_to_content, but leaves EXTRA margin on the posterior (+X, axis 2) side
    so a muscle layer can be added behind the breast after cropping.
    The bounding box is taken from the FIRST array only (the label), since the MRI image
    is nonzero almost everywhere and would prevent any cropping."""
    nz = (arrs[0] != 0)
    cz, cy, cx = np.where(nz)
    z0, z1 = max(0, cz.min()-pad), min(arrs[0].shape[0], cz.max()+pad+1)
    y0, y1 = max(0, cy.min()-pad), min(arrs[0].shape[1], cy.max()+pad+1)
    x0      = max(0, cx.min()-pad)
    x1      = min(arrs[0].shape[2], cx.max()+back_pad_x+1)   # extra room behind breast
    slices = (slice(z0,z1), slice(y0,y1), slice(x0,x1))
    return [a[slices] for a in arrs], slices
