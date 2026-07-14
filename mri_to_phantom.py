from __future__ import annotations
import argparse
from pathlib import Path
import sys
import numpy as np
import nibabel as nib
import torch
import torchio as tio
import pydicom
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from tqdm import tqdm
import torch.nn.functional as F
from itertools import permutations, product
from nibabel.orientations import io_orientation, axcodes2ornt, ornt_transform, apply_orientation
import preprocessing   # per-package image preprocessing (z-score always; denoise baked per package)

# Duke fixed hyperparameters
BREAST_INPUT_DIM = (144, 144, 96)
DV_INPUT_DIM = 96
DV_X_Y_DIVS = 8
DV_Z_DIVS = 3

SCRIPT_DIR = Path(__file__).resolve().parent
DUKE_REPO = SCRIPT_DIR / "3D-Breast-FGT-and-Blood-Vessel-Segmentation-main"
DUKE_DATA_ROOT = SCRIPT_DIR / "data" / "Duke-Breast-Cancer-MRI"
MAMA_ROOT = SCRIPT_DIR / "data" / "mamamia_duke"
DEFAULT_OUT = SCRIPT_DIR / "outputs"

NNUNET_MODEL_DIR = SCRIPT_DIR / "mamamia_weights" / "nnUNet_results" / "full_image_dce_mri_tumor_segmentation"
_PREDICTOR = None  # lazy-loaded MAMA-MIA nnU-Net (only built if an expert seg is missing)

# =========================================================
# PREPROCESSING
# =========================================================
def normalize_image(arr, min_cutoff=0.001, max_cutoff=0.001):
    a = arr.astype(np.float32)
    lo = float(np.quantile(a, min_cutoff))
    hi = float(np.quantile(a, 1.0 - max_cutoff))
    a = np.clip(a, lo, hi)
    return (a - lo) / max(hi - lo, 1e-8)

def zscore_image(arr):
    a = arr.astype(np.float32)
    return (a - float(a.mean())) / max(float(a.std()), 1e-8)

def duke_preprocess(arr):
    return zscore_image(normalize_image(arr))

# =========================================================
# LABEL MAP
# =========================================================
def build_label_map(breast, dv_probs, tumor, fgt_threshold=0.20, fold_vessels=True):
    label = np.zeros_like(breast, dtype=np.uint8)
    inside = (breast > 0)
    label[inside] = 1
    is_fgt = (dv_probs[2] > fgt_threshold) | (dv_probs[1] > fgt_threshold) if fold_vessels \
             else (dv_probs[2] > fgt_threshold)
    label[inside & is_fgt] = 2
    label[tumor > 0] = 3   # tumor always assigned, never gated by breast mask
    return label

# =========================================================
# MODEL LOADING & INFERENCE
# =========================================================
def _strip_module_prefix(state):
    return {(k[len("module."):] if k.startswith("module.") else k): v for k, v in state.items()}

def load_duke_models(duke_dir, device):
    sys.path.insert(0, duke_dir)
    from unet import UNet3D
    breast = UNet3D(in_channels=1, out_classes=1, num_encoding_blocks=3,
                    padding=True, normalization="batch").to(device)
    breast.load_state_dict(_strip_module_prefix(
        torch.load(f"{duke_dir}/trained_models/breast_model.pth", map_location=device)))
    breast.eval()
    dv = UNet3D(in_channels=2, out_classes=3, num_encoding_blocks=3,
                padding=True, normalization="batch").to(device)
    dv.load_state_dict(_strip_module_prefix(
        torch.load(f"{duke_dir}/trained_models/dv_model.pth", map_location=device)))
    dv.eval()
    return breast, dv

def _box_starts(L, n, sub):
    if n <= 1: return [0]
    step = (L - sub) // (n - 1)
    return [i * step for i in range(n - 1)] + [L - sub]

def predict_fgt_argmax(image, breast_prob, model, device,
                       sub=DV_INPUT_DIM, nx=DV_X_Y_DIVS, ny=DV_X_Y_DIVS, nz=DV_Z_DIVS,
                       return_probs=False):
    img_norm = duke_preprocess(image)
    msk = np.clip(breast_prob.astype(np.float32), 0.0, 1.0)
    X, Y, Z = img_norm.shape
    pad = (max(0, sub-X), max(0, sub-Y), max(0, sub-Z))
    img_pad = np.pad(img_norm, ((0,pad[0]),(0,pad[1]),(0,pad[2])))
    msk_pad = np.pad(msk,     ((0,pad[0]),(0,pad[1]),(0,pad[2])))
    Xp,Yp,Zp = img_pad.shape
    sx,sy,sz = _box_starts(Xp,nx,sub), _box_starts(Yp,ny,sub), _box_starts(Zp,nz,sub)
    sum_probs = np.zeros((3,Xp,Yp,Zp), dtype=np.float32)
    counts    = np.zeros((Xp,Yp,Zp),   dtype=np.int32)
    for x0,y0,z0 in tqdm([(x0,y0,z0) for x0 in sx for y0 in sy for z0 in sz],
                          desc="dv tiles", leave=False):
        inp = np.stack([img_pad[x0:x0+sub,y0:y0+sub,z0:z0+sub],
                        msk_pad[x0:x0+sub,y0:y0+sub,z0:z0+sub]], axis=0)
        with torch.no_grad():
            probs = F.softmax(model(torch.from_numpy(inp).unsqueeze(0).float().to(device)), dim=1)
        sum_probs[:,x0:x0+sub,y0:y0+sub,z0:z0+sub] += probs.squeeze(0).cpu().numpy()
        counts[x0:x0+sub,y0:y0+sub,z0:z0+sub] += 1
    avg = (sum_probs / np.maximum(counts,1)[np.newaxis])[:,:X,:Y,:Z]
    return avg if return_probs else np.argmax(avg, axis=0).astype(np.uint8)

def predict_breast_prob(image, model, device):
    img_norm = duke_preprocess(image)
    img_t = torch.from_numpy(img_norm).float().unsqueeze(0)
    subj = tio.Subject(image=tio.ScalarImage(tensor=img_t))
    inp = tio.Resize(BREAST_INPUT_DIM)(subj).image.tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        out = torch.sigmoid(model(inp)).squeeze(0).cpu()
    prob = tio.Resize(image.shape)(tio.Subject(image=tio.ScalarImage(tensor=out))).image.tensor.squeeze(0).numpy()
    return np.clip(prob.astype(np.float32), 0.0, 1.0)

# =========================================================
# DICOM LOADING + RAS REORIENTATION
# =========================================================
def find_precontrast_series(study_dir):
    EXCLUDE = ["ph1","ph2","ph3","ph4","+c","post","segmentation"]
    INCLUDE = ["ax dyn pre","pre contrast","pre_contrast","dyn pre","ax 3d dyn","t1 pre","t1"]
    candidates = [d for d in study_dir.iterdir() if d.is_dir()
                  and not any(ex in d.name.lower() for ex in EXCLUDE)]
    for kw in INCLUDE:
        matches = [d for d in candidates if kw in d.name.lower()]
        if matches:
            print(f"  Using series: {matches[0].name!r} (matched '{kw}')")
            return matches[0]
    if candidates:
        best = max(candidates, key=lambda d: len(list(d.glob("*.dcm"))))
        print(f"  WARNING: falling back to {best.name!r}")
        return best
    raise FileNotFoundError(f"No pre-contrast series in {study_dir}")

def load_full_dicom_series(patient_id, duke_data_root):
    patient_dir = duke_data_root / patient_id
    study_dirs = [d for d in patient_dir.iterdir() if d.is_dir()
                  and "MRI BREAST" in d.name.upper()] or list(patient_dir.iterdir())
    series_dir = find_precontrast_series(study_dirs[0])
    dcm_files = sorted(series_dir.glob("*.dcm"))
    slices = sorted([pydicom.dcmread(f) for f in dcm_files],
                    key=lambda x: float(x.ImagePositionPatient[2]))
    first = slices[0]
    vol = np.zeros((int(first.Rows), int(first.Columns), len(slices)), dtype=np.float32)
    for i, ds in enumerate(slices):
        vol[:, :, i] = ds.pixel_array
    return vol, first

def build_dicom_affine(first_slice):
    iop = [float(x) for x in first_slice.ImageOrientationPatient]
    ipp = [float(x) for x in first_slice.ImagePositionPatient]
    ps  = [float(first_slice.PixelSpacing[0]), float(first_slice.PixelSpacing[1])]
    st  = float(first_slice.SliceThickness)
    row = np.array(iop[0:3]); col = np.array(iop[3:6]); slc = np.cross(row, col)
    aff = np.eye(4)
    aff[:3,0] = row * ps[0]
    aff[:3,1] = col * ps[1]
    aff[:3,2] = slc * st
    aff[:3,3] = ipp
    return aff

def reorient_to_ras(volume, affine):
    """Reorient a (rows, cols, slices) volume to RAS using its DICOM affine."""
    cur = nib.aff2axcodes(affine)
    if cur == ('R', 'A', 'S'):
        print(f"  DICOM already RAS — no reorientation needed")
        return volume
    print(f"  Reorienting DICOM from {cur} to RAS")
    transform = ornt_transform(io_orientation(affine), axcodes2ornt('RAS'))
    return apply_orientation(volume, transform)





def find_first_postcontrast_series(study_dir):
    """First POST-contrast series = model input (DUKE '1st pass', ISPY2 'Ph1'/601)."""
    def score(d):
        n = d.name.lower()
        if "1st pass" in n or "ph1" in n: return 0
        if "pass" in n or "ph" in n:      return 1
        if "pre" in n:                    return 9
        return 5
    cand = [d for d in study_dir.iterdir() if d.is_dir()
            and any(k in d.name.lower() for k in ("pass", "ph", "dyn", "vibrant"))]
    if not cand:
        cand = [d for d in study_dir.iterdir() if d.is_dir()]
    best = sorted(cand, key=score)[0]
    print(f"  Model input series: {best.name!r}")
    return best

def _load_series_array(series_dir):
    dcm = sorted(series_dir.glob("*.dcm"))
    sl = sorted([pydicom.dcmread(f) for f in dcm], key=lambda x: float(x.ImagePositionPatient[2]))
    first = sl[0]
    vol = np.zeros((int(first.Rows), int(first.Columns), len(sl)), np.float32)
    for i, ds in enumerate(sl):
        vol[:, :, i] = ds.pixel_array
    return vol, first

def _get_predictor():
    global _PREDICTOR
    if _PREDICTOR is None:
        import torch
        _ol = torch.load
        torch.load = lambda *a, **k: _ol(*a, **{**k, "weights_only": False})
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
        p = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=True,
                            perform_everything_on_device=torch.cuda.is_available(),
                            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                            verbose=False, allow_tqdm=True)
        folds = tuple(range(len([d for d in NNUNET_MODEL_DIR.iterdir()
<<<<<<< HEAD
                                 if d.name.startswith("fold_")])))
=======
                                if d.name.startswith("fold_")])))
>>>>>>> 95a62d9278519d9f9f3cfd30ad3bb5ac86b4f789
        # folds = (0,)      # single fold — avoids the CPU accumulate bug, ~5x faster on CPU
        p.initialize_from_trained_model_folder(str(NNUNET_MODEL_DIR), use_folds=folds,
                                               checkpoint_name="checkpoint_final.pth")
        _PREDICTOR = p
    return _PREDICTOR

def segment_tumor_with_model(patient_id, img_ras, dicom_aff):
    """Run MAMA-MIA nnU-Net on the RAW first post-contrast DICOM series; return tumor on RAS grid."""
    import tempfile, shutil
    patient_dir = DUKE_DATA_ROOT / patient_id
    study_dirs = [d for d in patient_dir.iterdir() if d.is_dir()
                  and "MRI BREAST" in d.name.upper()] or list(patient_dir.iterdir())
    post_dir = find_first_postcontrast_series(study_dirs[0])
    post_native, _ = _load_series_array(post_dir)
    tmp = tempfile.mkdtemp()
    try:
        inp = f"{tmp}/{patient_id}_0000.nii.gz"
        nib.Nifti1Image(post_native, dicom_aff).to_filename(inp)
        out = f"{tmp}/{patient_id}.nii.gz"
        _get_predictor().predict_from_files([[inp]], [out], save_probabilities=False,
                                            overwrite=True, num_processes_preprocessing=2,
                                            num_processes_segmentation_export=1)
        tum_can = nib.as_closest_canonical(nib.load(out))
        tum_can = nib.Nifti1Image(np.asarray(tum_can.get_fdata()), tum_can.affine, tum_can.header)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    tumor_ras = (tum_can.get_fdata() > 0.5).astype(np.uint8)
    assert tumor_ras.shape == img_ras.shape, f"tumor {tumor_ras.shape} != img_ras {img_ras.shape}"
    print(f"  Model tumor voxels (RAS): {int(tumor_ras.sum())}")
    return tumor_ras



# =========================================================
# TUMOR ALIGNMENT — matched against the RAS volume
# =========================================================
def align_tumor_to_ras(tumor_path, ras_vol, mama_image_path=None):
    """
    Align the MAMA-MIA tumor mask to the RAS-reoriented DICOM volume.

    The MAMA-MIA NIfTI and the Duke DICOM are the SAME scan but stored with a
    different axis order/flips. We find the exact transpose+flip that maps the
    MAMA *image* onto the RAS DICOM volume (by exact pixel match, with a
    correlation fallback), then apply that same transform to the tumor mask.
    """
    tumor_nii = nib.load(str(tumor_path))
    tumor_data = (tumor_nii.get_fdata() > 0.5).astype(np.uint8)
    print(f"  MAMA tumor: {nib.aff2axcodes(tumor_nii.affine)} orientation, shape {tumor_data.shape}")

    def make_apply(perm, fx, fy, fz):
        def fn(arr):
            t = np.transpose(arr, perm)
            if fx: t = np.flip(t, 0)
            if fy: t = np.flip(t, 1)
            if fz: t = np.flip(t, 2)
            return t
        return fn

    transform = None

    # Match the MAMA image to the RAS DICOM volume to discover the transform
    if mama_image_path is not None and Path(mama_image_path).exists():
        mama_img = nib.load(str(mama_image_path)).get_fdata().astype(np.float32)

        # Pass 1: exact pixel match
        for perm in permutations([0,1,2]):
            if np.transpose(mama_img, perm).shape != ras_vol.shape:
                continue
            for fx, fy, fz in product([False,True], repeat=3):
                if np.array_equal(make_apply(perm,fx,fy,fz)(mama_img), ras_vol):
                    transform = (perm, fx, fy, fz)
                    print(f"  Tumor transform (exact match): transpose{perm} flips(X={fx},Y={fy},Z={fz})")
                    break
            if transform: break

        # Pass 2: best correlation (handles intensity scaling / minor diffs)
        if transform is None:
            best_score, best = -1, None
            dref = (ras_vol - ras_vol.mean())
            dnorm = dref / (dref.std() + 1e-8)
            for perm in permutations([0,1,2]):
                if np.transpose(mama_img, perm).shape != ras_vol.shape:
                    continue
                for fx, fy, fz in product([False,True], repeat=3):
                    t = make_apply(perm,fx,fy,fz)(mama_img)
                    tn = (t - t.mean()) / (t.std() + 1e-8)
                    score = float(np.mean(dnorm * tn))   # normalized correlation
                    if score > best_score:
                        best_score, best = score, (perm,fx,fy,fz)
            transform = best
            print(f"  Tumor transform (best corr={best_score:.3f}): transpose{best[0]} flips{best[1:]}")
    else:
        print("  WARNING: MAMA image not found — using default transpose(1,0,2)+flipX")
        transform = ((1,0,2), True, False, False)

    perm, fx, fy, fz = transform
    tumor_ras = make_apply(perm, fx, fy, fz)(tumor_data).astype(np.uint8)

    # Resample if shape still differs (rare, only if grids truly mismatch)
    if tumor_ras.shape != ras_vol.shape:
        from scipy.ndimage import zoom
        zf = [ras_vol.shape[i]/tumor_ras.shape[i] for i in range(3)]
        tumor_ras = (zoom(tumor_ras.astype(np.float32), zf, order=0) > 0.5).astype(np.uint8)
        print(f"  Resampled tumor to {tumor_ras.shape}")

    # ── Safety check: tumor should sit on bright tissue, not background ──
    if tumor_ras.sum() > 0:
        px = ras_vol[tumor_ras > 0]
        on_tissue = float((px > ras_vol.mean()).mean())
        tc = np.where(tumor_ras > 0)
        print(f"  Tumor voxels: {tumor_ras.sum()}, centroid X={tc[0].mean():.0f}, Y={tc[1].mean():.0f}, Z={tc[2].mean():.0f}")
        print(f"  Tumor-on-tissue: {on_tissue:.0%} (intensity {px.mean():.0f} vs image mean {ras_vol.mean():.0f})")
        if on_tissue < 0.7:
            print(f"  ⚠️  WARNING: tumor mostly on background — alignment may be WRONG for this patient!")
    else:
        print("  ⚠️  WARNING: tumor empty after alignment!")

    return tumor_ras

# =========================================================
# PLOTTING
# =========================================================
def plot_comparison(img_vol, label_full, patient_name, out_dir, dpi=150):
    LABEL_CMAP = ListedColormap([
        (0,0,0,0), (1.0,0.85,0.4,0.55), (0.2,0.7,0.95,0.65), (0.95,0.15,0.15,0.85)])
    LABEL_NORM = BoundaryNorm([-0.5,0.5,1.5,2.5,3.5], 4)
    tv = np.where(label_full == 3)
    if len(tv[0]) > 0:
        c0,c1,c2 = int(tv[0].mean()), int(tv[1].mean()), int(tv[2].mean())
        title_note = f"TUMOR at X={c0}, Y={c1}, Z={c2}"
    else:
        c0,c1,c2 = [s//2 for s in img_vol.shape]; title_note = "NO TUMOR"
    img_disp = (img_vol - img_vol.min())/(img_vol.max()-img_vol.min()+1e-8)
    fig, axes = plt.subplots(2, 3, figsize=(24, 16))
    for j in range(3):
        sl_img = [img_disp[c0,:,:], img_disp[:,c1,:], img_disp[:,:,c2]][j]
        sl_lbl = [label_full[c0,:,:], label_full[:,c1,:], label_full[:,:,c2]][j]
        view = ["Sagittal","Coronal","Axial"][j]
        axes[0,j].imshow(sl_img, cmap='gray', origin='lower')
        axes[0,j].set_title(f'Original — {view}', fontsize=12); axes[0,j].axis('off')
        axes[1,j].imshow(sl_img, cmap='gray', origin='lower')
        axes[1,j].imshow(sl_lbl, cmap=LABEL_CMAP, norm=LABEL_NORM, origin='lower', alpha=0.6)
        axes[1,j].set_title(f'Segmented — {view}', fontsize=12); axes[1,j].axis('off')
    plt.suptitle(f'{patient_name} — {title_note}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    p = out_dir / f"{patient_name}_comparison.png"
    plt.savefig(p, dpi=dpi, bbox_inches='tight')
    print(f"✅ Saved plot: {p}")
    plt.show()

# =========================================================
# MAIN
# =========================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--patient',          type=str,   default='Breast_MRI_001')
    ap.add_argument('--mama-patient',     type=str,   default='DUKE_001')
    ap.add_argument('--breast-threshold', type=float, default=0.30)
    ap.add_argument('--fgt-threshold',    type=float, default=0.15)
    ap.add_argument('--no-tumor',         action='store_true')
    ap.add_argument('--device',           type=str,   default='auto',
                    choices=['auto','cuda','cpu','mps'])
    ap.add_argument('--out-dir',          type=str,   default='outputs')
    ap.add_argument('--no-save',          action='store_true')
    ap.add_argument('--no-plot',          action='store_true')
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Processing {args.patient} (MAMA: {args.mama_patient})")

    # 1. Load DICOM (native orientation)
    img_native, first_slice = load_full_dicom_series(args.patient, DUKE_DATA_ROOT)
    dicom_aff = build_dicom_affine(first_slice)
    print(f"  DICOM loaded: {img_native.shape}, native orientation {nib.aff2axcodes(dicom_aff)}")

    # 2. Reorient DICOM → RAS (models were trained on RAS)
    img_ras = reorient_to_ras(img_native, dicom_aff)
    print(f"  RAS volume: {img_ras.shape}")

    # 3. Preprocess for model AND saved image (denoise baked per package, then z-score)
    img_full = preprocessing.preprocess(img_ras).astype(np.float32)

    # 4. Device + models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") \
             if args.device == 'auto' else torch.device(args.device)
    print(f"  Device: {device}")
    breast_model, dv_model = load_duke_models(str(DUKE_REPO), device)

    # 5. Breast + FGT segmentation (on RAS image)
    prob_full   = predict_breast_prob(img_ras, breast_model, device)
    breast_full = (prob_full > args.breast_threshold).astype(np.uint8)
    print(f"  Breast voxels: {breast_full.sum()}")
    dv_full = predict_fgt_argmax(img_ras, prob_full, dv_model, device, return_probs=True)

    # 6. Tumor — aligned to the RAS volume
    if args.no_tumor:
        tumor_final = np.zeros_like(img_ras, dtype=np.uint8)
    else:
        tumor_path = MAMA_ROOT / "segmentations" / "expert" / f"{args.mama_patient}.nii.gz"
        mama_img_path = MAMA_ROOT / "images" / args.mama_patient / f"{args.mama_patient}_0000.nii.gz"
        if tumor_path.exists():
            tumor_final = align_tumor_to_ras(tumor_path, img_ras, mama_img_path)
        else:
            print(f"  No expert seg -> segmenting tumor with MAMA-MIA nnU-Net")
            tumor_final = segment_tumor_with_model(args.patient, img_ras, dicom_aff)

    # 7. Label map
    label_full = build_label_map(breast_full, dv_full, tumor_final, args.fgt_threshold)
    # label_full[label_full == 3] = 2 # consider tumor as fgt for test!
    print(f"\n  Label summary:")
    print(f"    Background:   {(label_full==0).sum()}")
    print(f"    Fatty tissue: {(label_full==1).sum()}")
    print(f"    FGT/vessels:  {(label_full==2).sum()}")
    print(f"    Tumor:        {(label_full==3).sum()}")

    # 8. Save (RAS image + label, with RAS affine carrying voxel spacing)
    if not args.no_save:
        po = out_dir / args.patient; po.mkdir(parents=True, exist_ok=True)
        # RAS affine from reoriented canonical image (keeps spacing + RAS directions)
        ras_affine = nib.as_closest_canonical(nib.Nifti1Image(img_native, dicom_aff)).affine
        nib.Nifti1Image(img_full.astype(np.float32), ras_affine).to_filename(po / f"{args.patient}_image.nii.gz")
        nib.Nifti1Image(label_full.astype(np.uint8), ras_affine).to_filename(po / f"{args.patient}_label.nii.gz")
        print(f"✅ Saved to: {po}")

    # 9. Plot
    if not args.no_plot:
        po = out_dir / args.patient; po.mkdir(parents=True, exist_ok=True)
        plot_comparison(img_ras, label_full, args.patient, po)

    print("\n✅ DONE!")

if __name__ == "__main__":
    main()
