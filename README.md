# Breast MRI -> dielectric phantom pipeline

**Preprocessing (this package):** z-score + SWITCHING median (cross footprint, k=3): edge-preserving impulse removal.


## Pipeline
1. `mri_to_phantom.py` — DICOM -> RAS -> preprocess -> Duke breast/FGT segmentation + tumour alignment -> image+label NIfTI
2. `split_muscle.py` — split breasts, add contoured muscle + skin shell
3. `dielectric_methods.py` — "ours" piecewise-linear Cole-Cole mapping; TWO versions: ours_segmentation, ours_gmm
4. `compare_dielectric.py` — seg-vs-gmm comparison plots
`full_pipeline.py` orchestrates everything.


## Run
    python full_pipeline.py --patient Breast_MRI_001 --mama-patient DUKE_001
    python full_pipeline.py --patient Breast_MRI_001 --skip-segmentation
Requires the Duke models, MAMA-MIA data, and DICOM under the paths in mri_to_phantom.py.
Outputs land in outputs/<patient>/.
