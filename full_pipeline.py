#!/usr/bin/env python
"""
full_pipeline.py — End-to-end breast MRI -> dielectric phantom pipeline ("ours" only).

Per patient, in order:
  1. MRI -> segmentation (raw DICOM -> image + label, RAS)        [mri_to_phantom.py]
     (image preprocessing — z-score, baked per package — is applied INSIDE mri_to_phantom,
      so segmentation AND dielectric mapping use the same preprocessed image)
  2. Split breasts + add muscle + skin                            [split_muscle.py]
  3. Dielectric mapping, "ours" piecewise-linear, TWO versions:
        ours_segmentation, ours_gmm   at 3, 6, 10 GHz
  4. Comparison plots (seg vs gmm)
  5. Save everything under outputs/<patient>/

PERFORMANCE: the intensity breakpoints + sub-tissue bands are FREQUENCY-INDEPENDENT, so
they (and the GMM) are computed ONCE per source via dm.prepare_breaks() and reused across
all frequencies.  Skylar removed; only "ours" remains.

Usage:
    python full_pipeline.py --patient Breast_MRI_001 --mama-patient DUKE_001
    python full_pipeline.py --patient Breast_MRI_001 --skip-segmentation
"""
import argparse, sys
from pathlib import Path
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import dielectric_methods as dm
import compare_dielectric as cmp
import split_muscle as sm

FREQS = [0.5, 0.75, 1.0, 2.0, 3.0]
ADD_SKIN = True
SKIN_THICKNESS_MM = 1.5
TUMOR_PERCENTILE = True
MUSCLE_THICKNESS_MM = 10.0
MUSCLE_LABEL = dm.MUSCLE

VERSIONS = [("seg", "ours_segmentation"), ("gmm", "ours_gmm")]

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
# STAGE 3+4 — dielectric mapping (2 versions, cached breakpoints) + plots
# =====================================================================
def process_breast(breast, patient, out_dir):
    name = breast["name"]; vol = breast["vol"].astype(np.float64); label = breast["label"]
    spacing = breast.get("spacing", (1.0,1.0,1.0))
    bdir = out_dir / patient / name; bdir.mkdir(parents=True, exist_ok=True)
    aff = np.diag([spacing[0], spacing[1], spacing[2], 1.0])
    nib.Nifti1Image(vol.astype(np.float32), aff).to_filename(bdir/f"{name}_image.nii.gz")
    nib.Nifti1Image(label, aff).to_filename(bdir/f"{name}_label.nii.gz")

    # ---- compute breakpoints + bands ONCE per source (the expensive GMM runs once) ----
    prep = dm.prepare_breaks(vol, label, sources=("seg","gmm"))

    per_freq = {}
    for f in FREQS:
        results = {}
        for src, vname in VERSIONS:
            x_break = prep[src]["x_break"]
            er, ei = dm.convert_ours(vol, label, x_break, f)        # frequency-dependent only
            if TUMOR_PERCENTILE: dm.assign_tumor_percentile(er, ei, label, vol, f)
            er, ei = dm.assign_muscle(er, ei, label, f)
            if ADD_SKIN: dm.assign_skin(er, ei, label, f, vol=vol)
            results[vname] = (er, ei)
            nib.Nifti1Image(er, aff).to_filename(bdir/f"{name}_{vname}_real_{f:g}GHz.nii.gz")
            nib.Nifti1Image(ei, aff).to_filename(bdir/f"{name}_{vname}_imag_{f:g}GHz.nii.gz")
        results["subt_seg"] = prep["seg"]["subt"]
        results["subt_gmm"] = prep["gmm"]["subt"]
        per_freq[f] = results
        print(f"    {name} @ {f:g} GHz done (ours_segmentation, ours_gmm)")
    breast["results"] = per_freq
    return per_freq

# =====================================================================
# MAIN
# =====================================================================
def main():
    global MUSCLE_THICKNESS_MM
    ap = argparse.ArgumentParser()
    ap.add_argument("--patient", required=True)
    ap.add_argument("--mama-patient", default=None)
    ap.add_argument("--out-dir", default=str(SCRIPT_DIR/"outputs"))
    ap.add_argument("--skip-segmentation", action="store_true")
    ap.add_argument("--muscle-thickness-mm", type=float, default=MUSCLE_THICKNESS_MM)
    args = ap.parse_args()
    MUSCLE_THICKNESS_MM = args.muscle_thickness_mm
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    print(f"=== {args.patient} ===  (muscle {MUSCLE_THICKNESS_MM} mm)")

    print("[1/4] segmentation (preprocessing baked in mri_to_phantom)")
    if not args.skip_segmentation:
        run_segmentation(args.patient,
                         args.mama_patient or "DUKE_"+args.patient.split("_")[-1], out_dir)

    print("[2/4] split breasts + add muscle")
    breasts = split_and_add_muscle(args.patient, out_dir)
    print(f"   -> {len(breasts)} breast(s)")

    print("[3/4 & 4/4] dielectric mapping (2 versions) + comparison plots")
    for b in breasts:
        process_breast(b, args.patient, out_dir)

    pdir = out_dir / args.patient
    for f in FREQS:
        bdata = [dict(name=b["name"], label=b["label"], results=b["results"][f]) for b in breasts]
        cmp.two_version_maps(bdata, f, pdir/f"{args.patient}_maps_2version_{f:g}GHz.png")
        cmp.two_version_distributions(bdata, f, pdir/f"{args.patient}_dist_real_2version_{f:g}GHz.png", part="real")
        cmp.two_version_distributions(bdata, f, pdir/f"{args.patient}_dist_imag_2version_{f:g}GHz.png", part="imag")
        print(f"   2-version plots @ {f} GHz saved")

    print(f"\n✅ DONE. Everything under {out_dir/args.patient}/")

if __name__ == "__main__":
    main()
