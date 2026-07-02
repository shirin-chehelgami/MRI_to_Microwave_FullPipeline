# Breast MRI -> dielectric phantom pipeline (switching preprocessing)

**Preprocessing (this package):** z-score + SWITCHING median (cross footprint, k=3): edge-preserving impulse removal. Removes specks while leaving boundaries and composition intact. RECOMMENDED.
Defined in `preprocessing.py` — the ONLY file that differs between the three packages. It is
applied inside `mri_to_phantom.py`, so segmentation AND dielectric mapping use the same image.

## Pipeline (run order)
1. `mri_to_phantom.py` — DICOM -> RAS -> preprocess -> Duke breast/FGT segmentation + tumour alignment -> image+label NIfTI
2. `split_muscle.py` — split breasts, add contoured muscle + skin shell
3. `dielectric_methods.py` — "ours" piecewise-linear Cole-Cole mapping; TWO versions: ours_segmentation, ours_gmm
4. `compare_dielectric.py` — seg-vs-gmm comparison plots
`full_pipeline.py` orchestrates everything.

## Changes vs the original
- Skylar removed — only "ours" (ours_segmentation, ours_gmm).
- Speed fix — breakpoints + GMM are frequency-independent, computed ONCE per source
  (dm.prepare_breaks) instead of ~12x per breast: ~4.7x faster (31.5s -> 6.7s, one breast, 3 freqs).
- [0,255] removed — z-score always applied (Duke-model requirement).

## Run
    python full_pipeline.py --patient Breast_MRI_001 --mama-patient DUKE_001
    python full_pipeline.py --patient Breast_MRI_001 --skip-segmentation
Requires the Duke models, MAMA-MIA data, and DICOM under the paths in mri_to_phantom.py.
Outputs land in outputs/<patient>/.
