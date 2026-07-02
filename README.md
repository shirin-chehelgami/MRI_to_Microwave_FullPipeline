# Breast MRI -> dielectric phantom pipeline

**Preprocessing (this package):** z-score + SWITCHING median (cross footprint, k=3): edge-preserving impulse removal.


## Pipeline
1. `mri_to_phantom.py` — DICOM -> RAS -> preprocess -> Duke breast/FGT segmentation + tumour alignment -> image+label NIfTI
2. `split_muscle.py` — split breasts, add contoured muscle + skin shell
3. `dielectric_methods.py` — "ours" piecewise-linear Cole-Cole mapping; TWO versions: ours_segmentation, ours_gmm
4. `compare_dielectric.py` — seg-vs-gmm comparison plots
`full_pipeline.py` orchestrates everything.


## Run
    python full_pipeline.py --patient Breast_MRI_001 --muscle-thickness-mm 0
    python full_pipeline.py --patient Breast_MRI_001 --skip-segmentation --muscle-thickness-mm 0
Requires the Duke models, MAMA-MIA data, and DICOM under the paths in mri_to_phantom.py.
Outputs land in outputs/<patient>/.


## Run
    python nii_to_chamber.py ./outputs/Breast_MRI_001/breast1/breast1_label.nii.gz   ./outputs/Breast_MRI_001/breast1/breast1_ours_segmentation_real_3GHz.nii.gz ./outputs/Breast_MRI_001/breast1/breast1_ours_segmentation_imag_3GHz.nii.gz ./outputs/Breast_MRI_001/breast1/breast1_in_chamber.mat

