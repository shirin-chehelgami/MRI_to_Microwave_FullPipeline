markdown

# MRI to Microwave Breast Data Processing Pipeline

A complete pipeline for breast MRI segmentation and dielectric property conversion for microwave imaging applications.

## Overview

This pipeline processes Duke breast MRI datasets through two main stages:
1. **Segmentation** - Uses Duke's pre-trained models to segment breast tissue (fat and FGT/fibroglandular tissue)
   
         Lew CO, Harouni M, Kirksey ER, Kang EJ, Dong H, Gu H, Grimm LJ, Walsh R, Lowell DA, Mazurowski MA. A publicly available deep learning model and dataset for segmentation of breast, fibroglandular tissue, and vessels in breast MRI. Sci Rep. 2024 Mar 5;14(1):5383. doi: 10.1038/s41598-024-54048-2. PMID: 38443410; PMCID: PMC10915139.
3. **Tumor Segmentation** - Uses Mamamia expert's tumor segmentation


         Garrucho, L., Kushibar, K., Reidel, CA. et al. A large-scale multicenter breast cancer DCE-MRI benchmark dataset with expert segmentations. Sci Data 12, 453 (2025). https://doi.org/10.1038/s41597-025-04707-4
5. **Dielectric Conversion** - Converts segmented tissue labels to frequency-dependent dielectric properties (ε′, ε″, σ)

   
         Zastrow E, Davis SK, Lazebnik M, Kelcz F, Van Veen BD, Hagness SC. Development of anatomically realistic numerical breast phantoms with accurate dielectric properties for modeling microwave interactions with the human breast. IEEE Trans Biomed Eng. 2008 Dec;55(12):2792-800. doi: 10.1109/TBME.2008.2002130. PMID: 19126460; PMCID: PMC2621084.

## Repository Structure

MRI_To_Microwave_Breast_Data/

├── mri_to_phantom.py # DICOM -> RAS -> preprocess -> Duke breast/FGT segmentation + tumour alignment -> image+label NIfTI

├── full_pipeline.py

├── compare_dielectric.py # seg-vs-gmm comparison plots

├── dielectric_methods.py # "ours" piecewise-linear Cole-Cole mapping; TWO versions: ours_segmentation, ours_gmm

├── split_muscle.py # split breasts, add contoured muscle + skin shell

├── preprocessing.py # z-score + SWITCHING median (cross footprint, k=3): edge-preserving impulse removal

├── nii_to_chamber.py # create voxel file for Forward Solver

├── 3D-Breast-FGT-and-Blood-Vessel-Segmentation-main/ # Duke's models for breast and FGT segmentation

├── data/

│ ├── mamamia_duke/ # MAMA-MIA duke dataset (tumor masks)

│ │ ├── images/

│ │ └── segmentations/

│ ├── Duke-Breast-Cancer-MRI/ # Original duke dataset

│── Breast_MRI_001/

├── Breast_MRI_002/

└── ...
└── outputs/ # All results saved here

├── Breast_MRI_001/

├── Breast_MRI_002/

└── ...
text


## Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU 

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/MRI_To_Microwave_FullPipeline.git
cd MRI_To_Microwave_Breast_Data
```

### 2. Install Dependencies
```bash

pip install -r requirements.txt
```




## Dataset Download Instructions

### Option 1: Download Full Dataset (Recommended)

The Duke Breast Cancer MRI dataset is available through The Cancer Imaging Archive (TCIA).

1. **Download NBIA Data Retriever** from: https://wiki.cancerimagingarchive.net/display/NBIA/Downloading+TCIA+Images

2. **Use the manifest file** provided in `data/manifest-1778260423557.tcia`
   - Double-click the `.tcia` file
   - It will open with NBIA Data Retriever
   - Select your download destination
   - Click "Start" to download all images

3. **Organize downloaded data** into `data/Duke-Breast-Cancer-MRI/`

### Option 2: Use Sample Data (Quick Test)

The repository includes **10 sample patients** already in `data/Duke-Breast-Cancer-MRI/` for testing. To get the full dataset, follow Option 1 above.



## Run

    python full_pipeline.py --patient Breast_MRI_001 --muscle-thickness-mm 0
    python full_pipeline.py --patient Breast_MRI_001 --skip-segmentation --muscle-thickness-mm 0

* Change FREQS = [3.0] in full_pipeline.py for more frequencies


## Run
    python nii_to_chamber.py ./outputs/Breast_MRI_001/breast1/breast1_label.nii.gz   ./outputs/Breast_MRI_001/breast1/breast1_ours_segmentation_real_3GHz.nii.gz ./outputs/Breast_MRI_001/breast1/breast1_ours_segmentation_imag_3GHz.nii.gz ./outputs/Breast_MRI_001/breast1/breast1_in_chamber.mat

