#!/usr/bin/env python
"""
full_pipeline.py — End-to-end breast MRI -> dielectric phantom pipeline

In order:
  1. MRI -> segmentation (raw DICOM -> image + label, RAS)        [mri_to_phantom.py]
     (image preprocessing — z-score, baked per package — is applied INSIDE mri_to_phantom,
      so segmentation AND dielectric mapping use the same preprocessed image)
  2. Split breasts + add muscle + skin                            [split_muscle.py]
  3. Dielectric mapping, "ours" piecewise-linear, TWO versions:
        ours_segmentation, ours_gmm
  4. Comparison plots (seg vs gmm)
  5. Save everything under outputs/<patient>/

 
Usage:
    python full_pipeline.py --patient 001 --mama-patient DUKE_001
    python full_pipeline.py --batch
    python full_pipeline.py --patient 001 002 --breast 1
    python full_pipeline.py --patient 001 --freq 1.3
    python full_pipeline.py --patient 001 --freq 0.5:3.0:0.01
    python full_pipeline.py --patient 001 --versions seg        # skip gmm
"""
import argparse, sys
from pathlib import Path
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
 
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
 
import preprocessing as pp
N4_MODE = "edge"          # None | "whole" | "edge"
N4_FEATHER_MM = 30.0    # edge mode only
 
 
import dielectric_methods as dm
import compare_dielectric as cmp
import split_muscle as sm
 
# default frequency sweep (used when --freq is not given)
FREQS = np.arange(0.5, 3.0, 0.01)
 
ADD_SKIN = True
SKIN_THICKNESS_MM = 2.0
TUMOR_PERCENTILE = True
MUSCLE_THICKNESS_MM = 10.0
MUSCLE_LABEL = dm.MUSCLE
 
VERSIONS = [("seg", "ours_segmentation"), ("gmm", "ours_gmm")]
 
 
# =====================================================================
# SHARED FLAG HELPERS (identical across the three scripts)
# =====================================================================
def parse_freqs(freq_args):
    """None -> use script default; 'a:b:s' -> arange; else explicit list."""
    if not freq_args:
        return None
    if len(freq_args) == 1 and ":" in freq_args[0]:
        a, b, s = (float(x) for x in freq_args[0].split(":"))
        return np.arange(a, b, s)
    return np.array([float(x) for x in freq_args])
 
def norm_patient(p):
    """accept '001' or 'Breast_MRI_001' -> 'Breast_MRI_001'."""
    return p if p.startswith("Breast_MRI_") else f"Breast_MRI_{p}"
 
def norm_breast(b):
    """accept '1' or 'breast1' -> 'breast1'."""
    return b if str(b).startswith("breast") else f"breast{b}"
 
 
# =====================================================================
# STAGE 1 — segmentation (preprocessing is applied inside mri_to_phantom)
# =====================================================================
def run_segmentation(patient, mama_patient, out_dir):
    import mri_to_phantom as seg
    sys.argv = ["mri_to_phantom.py", "--patient", patient, "--mama-patient", mama_patient,
                "--out-dir", str(out_dir), "--no-plot"]
    seg.main()
 
def load_saved(patient, out_dir):
    pdir = out_dir / patient
    img = nib.load(str(pdir / f"{patient}_image.nii.gz"))
    lab = nib.load(str(pdir / f"{patient}_label.nii.gz"))
    return img.get_fdata().astype(np.float32), lab.get_fdata().astype(np.uint8), img.affine
 
 
# =====================================================================
# STAGE 2 — split breasts + add muscle (validated build_phantom logic)
# =====================================================================
def split_and_add_muscle(patient, out_dir):
    pdir = out_dir / patient
    img, pd  = sm.read_nii_gz(pdir / f"{patient}_image.nii.gz")
    label, _ = sm.read_nii_gz(pdir / f"{patient}_label.nii.gz")
    label = label.astype(np.uint8); img = img.astype(np.float32)
 
    spacing_zyx = (float(pd[3]), float(pd[2]), float(pd[1]))
    print(f"   voxel spacing (Z,Y,X) = {tuple(round(s,4) for s in spacing_zyx)} mm")
    muscle_vox = max(0, int(round(MUSCLE_THICKNESS_MM / spacing_zyx[2])))
    skin_spacing = 0.5*(spacing_zyx[1]+spacing_zyx[2])
    skin_vox = max(1, int(round(SKIN_THICKNESS_MM / skin_spacing)))
    print(f"   muscle {MUSCLE_THICKNESS_MM} mm -> {muscle_vox} vox; skin {SKIN_THICKNESS_MM} mm -> {skin_vox} vox")
 
    breast_mask = (label > 0)
    split_y, right_c, left_c, mid_x = sm.find_split_matlab_exact(breast_mask)
    b1_end, b2_start, valley = sm.find_inner_edges(breast_mask, right_c, left_c)
    mask_b1 = breast_mask.copy(); mask_b1[:, b2_start:, :] = False
    mask_b2 = breast_mask.copy(); mask_b2[:, :b1_end + 1, :] = False
    thick_vox = muscle_vox
 
    results = []
    for name, mask_b in [("breast1", mask_b1), ("breast2", mask_b2)]:
        lab_b = np.where(mask_b, label, 0).astype(np.uint8)
        vol_b = img.copy()
        back_pad = thick_vox + 3
        cropped, _ = sm.crop_to_content_asym(lab_b, vol_b, pad=5, back_pad_x=back_pad)
        lab_c, vol_c = cropped
        mask_c = (lab_c > 0)
        muscle_c = sm.add_muscle_contour(mask_c, thickness_vox=thick_vox) & (lab_c == 0)
        lab_c[muscle_c] = MUSCLE_LABEL
        if ADD_SKIN:
            lab_c = dm.segment_skin(lab_c, thickness_vox=skin_vox)
        results.append(dict(name=name, vol=vol_c, label=lab_c, spacing=spacing_zyx))
    return results
 
 
# =====================================================================
# STAGE 3+4 — dielectric mapping (selected versions, cached breakpoints) + plots
# =====================================================================
def process_breast(breast, patient, out_dir):
    name = breast["name"]; vol = breast["vol"].astype(np.float64); label = breast["label"]
    vol_orig = vol.copy()
    spacing = breast.get("spacing", (1.0,1.0,1.0))
    bdir = out_dir / patient / name; bdir.mkdir(parents=True, exist_ok=True)
    aff = np.diag([spacing[0], spacing[1], spacing[2], 1.0])
 
    if N4_MODE == "whole":
            vol = pp.apply_n4_whole(vol, (label > 0)).astype(np.float64)
    elif N4_MODE == "edge":
        vol = pp.apply_n4_edge(vol, label, spacing, feather_mm=N4_FEATHER_MM).astype(np.float64)
    if N4_MODE:                              # clean the specks N4's division amplifies
        vol = pp.switching_median(vol).astype(np.float64)
 
    nib.Nifti1Image(vol.astype(np.float32), aff).to_filename(bdir/f"{name}_image.nii.gz")
    nib.Nifti1Image(label, aff).to_filename(bdir/f"{name}_label.nii.gz")
 
    # ---- compute breakpoints + bands ONCE, only for the selected sources ----
    srcs = tuple(s for s, _ in VERSIONS)             # e.g. ("seg",) or ("seg","gmm")
    prep = dm.prepare_breaks(vol, label, sources=srcs)
 
    per_freq = {}
    plotted_sample = False
    for f in FREQS:
        results = {}
        for src, vname in VERSIONS:
            x_break = prep[src]["x_break"]
            er, ei = dm.convert_ours(vol, label, x_break, f)        # frequency-dependent only
            if TUMOR_PERCENTILE: dm.assign_tumor_percentile(er, ei, label, vol, f)
            er, ei = dm.assign_muscle(er, ei, label, f)
            if ADD_SKIN: dm.assign_skin(er, ei, label, f, vol=vol)
            results[vname] = (er, ei)
            if not plotted_sample and src == "seg":
                            cmp.mri_vs_permittivity(vol_orig, vol, er, bdir/f"{name}_mri_vs_eps_sample_{f:g}GHz.png")
                            # cmp.mri_vs_permittivity(vol, er, bdir/f"{name}_preprocessed_mri_vs_eps_sample_{f:g}GHz.png")
                            plotted_sample = True
            print(f"  === {name} @ {f:g} GHz ({vname}) ===")
            print(f"    er:  min={er.min():.3f}  max={er.max():.3f}  nan={np.isnan(er).sum()}")
            print(f"    ei:  min={ei.min():.3f}  max={ei.max():.3f}  nan={np.isnan(ei).sum()}")
            print(f"    Negative ei: {(ei < 0).sum()}")
 
            if np.any(ei < 0):
                print("    !!! NEGATIVE IMAGINARY PART DETECTED !!!")
 
            nib.Nifti1Image(er, aff).to_filename(bdir/f"{name}_{vname}_real_{f:g}GHz.nii.gz")
            nib.Nifti1Image(ei, aff).to_filename(bdir/f"{name}_{vname}_imag_{f:g}GHz.nii.gz")
 
        for s in srcs:
            results[f"subt_{s}"] = prep[s]["subt"]
        per_freq[f] = results
        print(f"    {name} @ {f:g} GHz done ({', '.join(v for _, v in VERSIONS)})")
    breast["results"] = per_freq
    return per_freq
 
 
def run_one_patient(patient, args, out_dir):
    print(f"=== {patient} ===  (muscle {MUSCLE_THICKNESS_MM} mm)")
 
    print("[1/4] segmentation (preprocessing baked in mri_to_phantom)")
    if not args.skip_segmentation:
        run_segmentation(patient,
                         args.mama_patient or "DUKE_"+patient.split("_")[-1], out_dir)
 
    print("[2/4] split breasts + add muscle")
    breasts = split_and_add_muscle(patient, out_dir)
 
    # --breast filter
    if args.breast:
        keep = {norm_breast(b) for b in args.breast}
        breasts = [b for b in breasts if b["name"] in keep]
        if not breasts:
            print(f"   no breast matching {sorted(keep)}; skipping"); return
    print(f"   -> {len(breasts)} breast(s)")
 
    print("[3/4 & 4/4] dielectric mapping + comparison plots")
    for b in breasts:
        process_breast(b, patient, out_dir)
 
    pdir = out_dir / patient
    # comparison plots only make sense when BOTH versions were produced
    if len(VERSIONS) == 2:
        for f in FREQS:
            bdata = [dict(name=b["name"], label=b["label"], results=b["results"][f]) for b in breasts]
            cmp.two_version_maps(bdata, f, pdir/f"{patient}_maps_2version_{f:g}GHz.png")
            cmp.two_version_distributions(bdata, f, pdir/f"{patient}_dist_real_{f:g}GHz.png", part="real")
            cmp.two_version_distributions(bdata, f, pdir/f"{patient}_dist_imag_{f:g}GHz.png", part="imag")
            print(f"   2-version plots @ {f} GHz saved")
    else:
        print("   (single version selected — skipping 2-version comparison plots)")
 
    print(f"\n DONE. Everything under {out_dir/patient}/")
 
 
# =====================================================================
# MAIN
# =====================================================================
def main():
    global MUSCLE_THICKNESS_MM, FREQS, VERSIONS
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", action="store_true",
                    help="process ALL patients found under out-dir")
    ap.add_argument("--patient", nargs="+", default=None,
                    help="one or more patients, e.g. 001 002  (or Breast_MRI_001)")
    ap.add_argument("--breast", nargs="+", default=None,
                    help="one or both breasts, e.g. 1 2  (or breast1)")
    ap.add_argument("--freq", nargs="+", default=None,
                    help="list '1.0 1.3' OR range 'start:stop:step' e.g. 0.5:3.0:0.01")
    ap.add_argument("--versions", nargs="+", default=["seg", "gmm"],
                    choices=["seg", "gmm"],
                    help="which dielectric versions to produce (default both)")
    ap.add_argument("--mama-patient", default=None)
    ap.add_argument("--out-dir", default=str(SCRIPT_DIR/"outputs"))
    ap.add_argument("--skip-segmentation", action="store_true")
    ap.add_argument("--muscle-thickness-mm", type=float, default=MUSCLE_THICKNESS_MM)
    args = ap.parse_args()
 
    MUSCLE_THICKNESS_MM = args.muscle_thickness_mm
 
    # --versions filter
    VERSIONS = [(s, v) for (s, v) in VERSIONS if s in args.versions]
    if not VERSIONS:
        ap.error("--versions must include at least one of: seg, gmm")
 
    _f = parse_freqs(args.freq)
    if _f is not None:
        FREQS = _f
 
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
 
    # decide patient list
    if args.batch:
        import glob, os
        DATA_ROOT = "./data/Duke-Breast-Cancer-MRI"
        patients = sorted({os.path.basename(p) for p in
                           glob.glob(os.path.join(DATA_ROOT, "Breast_MRI_*"))})
        if not patients:
            print("no patients found for --batch"); return
    elif args.patient:
        patients = [norm_patient(p) for p in args.patient]
    else:
        ap.error("give --patient or --batch")
 
    for p in patients:
        print(f"\n########## {p} ##########")
        run_one_patient(p, args, out_dir)
 
 
if __name__ == "__main__":
    main()

