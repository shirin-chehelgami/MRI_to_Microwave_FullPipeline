#!/usr/bin/env python3
"""gen_configs.py — one GmshFEM .toml per patient/breast/frequency.

Usage:
    python3 gen_configs.py --batch                     # all patients, all breasts, all freqs
    python3 gen_configs.py --patient 001               # one patient
    python3 gen_configs.py --patient 001 002 --breast 1
    python3 gen_configs.py --patient 001 --freq 1.0 1.3
    python3 gen_configs.py --patient 001 --freq 0.5:3.0:0.01


For every matching  Breast_MRI_*/breast*  folder it:
  - reads depth_mm, diameter_mm from that breast's *_dimensions.csv
    (geometry is frequency-independent, computed once per breast)
  - finds every  *_chamber_*GHz.mat  and reads the frequency from the filename
  - writes one config.toml per (breast, frequency), mirroring the data tree:
        ./configs/Breast_MRI_001/breast1/config_1GHz.toml
"""
import os, glob, re, csv, sys, argparse
import numpy as np

ROOT   = "/Users/chehelgs-ins/Downloads/MRI_To_MW_Fullpipeline/outputs" # path to your duke pipeline
BASE   = "./examples/ExampleFD3DThinWire/ExampleFD3DThinWire.toml"   # path to your forward solver folder
CFGDIR = "./configs"


# ---- shared flag helpers (identical across all scripts) ----
def parse_freqs(freq_args):
    """None -> all; 'a:b:s' -> arange; else explicit list. Returns np.array or None."""
    if not freq_args:
        return None
    if len(freq_args) == 1 and ":" in freq_args[0]:
        a, b, s = (float(x) for x in freq_args[0].split(":"))
        return np.arange(a, b, s)
    return np.array([float(x) for x in freq_args])
 
def norm_breast(b):
    return b if str(b).startswith("breast") else f"breast{b}"


ap = argparse.ArgumentParser()
ap.add_argument("--batch", action="store_true")
ap.add_argument("--patient", nargs="+", default=None)   # e.g. 001 002
ap.add_argument("--breast",  nargs="+", default=None)   # e.g. 1 2  or breast1
ap.add_argument("--freq",    nargs="+", default=None)   # list or a:b:s
args = ap.parse_args()
 
# patient globs: explicit list, or "*" for batch/none
subj_globs = args.patient if args.patient else ["*"]
# breast globs
brst_globs = [norm_breast(b) for b in args.breast] if args.breast else ["breast*"]
# frequency filter set of "g"-formatted strings, or None = all
_f = parse_freqs(args.freq)
freq_filter = None if _f is None else { f"{v:g}" for v in _f }


with open(BASE) as f:
    template = f.read()

def setval(text, key, value):
    return re.sub(rf'(?m)^(\s*{re.escape(key)}\s*=\s*).*$', rf'\g<1>{value}', text, count=1)


# collect matching breast dirs across all patient/breast globs
dirs = []
for sg in subj_globs:
    for bg in brst_globs:
        dirs += glob.glob(os.path.join(ROOT, f"Breast_MRI_{sg}", bg))


made = 0
for d in sorted(glob.glob(os.path.join(ROOT, f"Breast_MRI_{SUBJ}", BRST))):
    subj = os.path.basename(os.path.dirname(d)).replace("Breast_MRI_", "")
    bn   = os.path.basename(d)
    tag  = f"{subj}_{bn}"

    # --- geometry (once per breast, frequency-independent) ---
    csvs = glob.glob(os.path.join(d, "*_dimensions.csv"))
    if not csvs:
        print(f"!! skip {tag}: no dimensions.csv"); continue
    with open(csvs[0]) as fh:
        row = list(csv.DictReader(fh))[0]
    depth_mm = float(row["depth_mm"])
    diam_mm  = float(row["diameter_mm"])
    phantom_radius     = abs(diam_mm) / 2 / 1000.0
    phantom_cyl_length = depth_mm / 1000.0 - phantom_radius

    # --- one config per chamber .mat found (frequency from the filename) ---
    chamber_mats = glob.glob(os.path.join(d, "*_chamber_*GHz.mat"))
    if not chamber_mats:
        print(f"!! skip {tag}: no chamber .mat"); continue

    for voxel in sorted(chamber_mats):
        m = re.search(r"_([\d.]+)GHz\.mat$", os.path.basename(voxel))
        if not m:
            continue
        ftag    = m.group(1) + "GHz"          # e.g. "1GHz" or "1.67GHz"
        freq_hz = float(m.group(1)) * 1e9

        out_path = f"./output/ExampleFD3DThinWire_{tag}_{ftag}"
        out_name = f"ExampleFD3DThinWire_{tag}_{ftag}"

        cfg = template
        cfg = setval(cfg, "voxel_file",         f'"{voxel}"')
        cfg = setval(cfg, "phantom_radius",     f"{phantom_radius:.6g}")
        cfg = setval(cfg, "phantom_cyl_length", f"{phantom_cyl_length:.6g}")
        cfg = setval(cfg, "output_path",        f'"{out_path}"')
        cfg = setval(cfg, "output_vtkfilename", f'"{out_name}"')
        cfg = setval(cfg, "freq_list",          f"[{freq_hz:g}]")

        subdir = os.path.join(CFGDIR, f"Breast_MRI_{subj}", bn)
        os.makedirs(subdir, exist_ok=True)
        with open(os.path.join(subdir, f"config_{ftag}.toml"), "w") as fh:
            fh.write(cfg)
        print(f"wrote {tag} @ {ftag}  r={phantom_radius:.4f}  L={phantom_cyl_length:.4f}")
        made += 1

print(f"\n{made} configs written to {CFGDIR}/")