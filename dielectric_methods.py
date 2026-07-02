"""
dielectric_methods.py — phantom -> dielectric mapping, "ours" (Zastrow piecewise-linear) only.

Two versions are produced downstream, differing ONLY in how the intensity breakpoints
are estimated:
    ours_segmentation : mu/sigma from labelled fat & fibro voxels
    ours_gmm          : mu/sigma from a 2-component GMM on the breast interior

PERFORMANCE: the 8 intensity breakpoints (and the GMM) depend only on (vol, label, source)
— NOT on frequency or version — so they are computed ONCE per source and reused across all
frequencies via `prepare_breaks()`.  (The old 4-way code refit the GMM ~12x per breast.)
"""
import numpy as np
from sklearn.mixture import GaussianMixture

EPS0 = 8.854e-12
PICO = 1e-12
GHZ  = 1e9

# ----- labels -----
FAT, FIBRO, TUMOR, MUSCLE = 1, 2, 3, 4
SKIN_LABEL = 5

# Table I single-pole Cole-Cole (eps_inf, d_eps, tau_ps, alpha, sigma)  [Lazebnik]
COLE_CURVES = {
 "maximum":(1.000,66.31,7.585,0.063,1.370),"glandular-high":(6.151,48.26,10.66,0.049,0.809),
 "glandular-median":(7.821,41.48,10.66,0.047,0.713),"glandular-low":(9.941,26.60,10.66,0.003,0.462),
 "fat-high":(4.031,3.654,14.12,0.055,0.083),"fat-median":(3.140,1.708,14.65,0.061,0.036),
 "fat-low":(2.908,1.200,16.88,0.069,0.020),"minimum":(2.293,0.141,16.40,0.251,0.002)}

# Pelicano Table 3/4 single-pole Debye (eps_inf, delta_eps, tau_ps, sigma)
MALIGNANT_DEBYE = {"p25": (12.9, 33.9, 13.0, 1.38), "p75": (14.6, 47.2, 13.0, 1.60)}
SKIN_DEBYE   = (15.93, 23.83, 13.0, 0.831)
MUSCLE_DEBYE = (21.66, 33.24, 13.0, 0.886)

# --- Gabriel 4-Cole-Cole (IFAC/CNR Appendix C), valid ~1 MHz - 20 GHz ---
# taus already converted to SECONDS: tau1*1e-12(ps), tau2*1e-9(ns), tau3*1e-6(us), tau4*1e-3(ms)
MUSCLE_GABRIEL = (
    4.0,                                                  # ef
    [50.0, 7000.0, 1.2e6, 2.5e7],                         # del1..4
    [7.234e-12, 353.678e-9, 318.310e-6, 2.274e-3],        # tau1..4 (seconds)
    [0.10, 0.10, 0.10, 0.00],                             # alf1..4
    0.2,                                                  # sigma (S/m)
)
SKIN_GABRIEL = (                                           # Gabriel "Skin (Dry)" — matches Pelicano
    4.0,                                                  # ef
    [32.0, 1100.0, 0.0, 0.0],                             # del1..4
    [7.234e-12, 32.481e-9, 159.155e-6, 15.915e-3],        # tau1..4 (seconds)
    [0.00, 0.20, 0.20, 0.20],                             # alf1..4
    0.0002,                                               # sigma (S/m)
)

CURVE_ORDER = ["minimum","fat-low","fat-median","fat-high",
               "glandular-low","glandular-median","glandular-high","maximum"]

def _cole_eval(f, p):
    ei,de,tp,al,sg = p; w = 2*np.pi*f; t = tp*PICO
    e = ei + de/(1+(1j*w*t)**(1-al)) + sg/(1j*w*EPS0); return e.real, -e.imag

def _debye_eval2(f, p):
    ei,de,tp,sg = p; w = 2*np.pi*f; t = tp*PICO
    e = ei + de/(1+1j*w*t) + sg/(1j*w*EPS0); return e.real, -e.imag

def _cole4_eval(f, p):
    """Gabriel 4-pole Cole-Cole. p = (ef, [del1..4], [tau1..4 in SECONDS], [alf1..4], sig)."""
    ef, dels, taus, alfs, sig = p
    w = 2*np.pi*f
    e = ef + sig/(1j*w*EPS0)
    for de, tau, al in zip(dels, taus, alfs):
        e = e + de/(1 + (1j*w*tau)**(1-al))
    return e.real, -e.imag

def _expected(pdf, lo, hi, cond):
    xg = np.linspace(lo,hi,20000); dx = xg[1]-xg[0]; pp = pdf(xg); m = cond(xg)
    return float(np.sum(xg[m]*pp[m]*dx)/np.sum(pp[m]*dx))

# =====================================================================
# Zastrow breakpoints (eq 1-8). FREQUENCY-INDEPENDENT — compute once per source.
# =====================================================================
def zastrow_breakpoints(vol, label, source="seg"):
    """source='seg': mu/sig from labelled fat & fibro voxels.
       source='gmm': mu/sig from a 2-component GMM on the breast interior."""
    interior = vol[(label==FAT)|(label==FIBRO)].astype(np.float64)
    if source == "gmm":
        X = interior.reshape(-1,1)
        if X.shape[0] > 300000:
            X = X[np.random.RandomState(0).choice(X.shape[0],300000,replace=False)]
        g = GaussianMixture(n_components=2, random_state=0).fit(X)
        gm=g.means_.ravel(); gs=np.sqrt(g.covariances_.ravel()); gw=g.weights_.ravel()
        o=np.argsort(gm)
        mu1,sig1,w1 = gm[o[0]],gs[o[0]],gw[o[0]]
        mu2,sig2,w2 = gm[o[1]],gs[o[1]],gw[o[1]]
    else:
        fat = vol[label==FAT].astype(np.float64); fib = vol[label==FIBRO].astype(np.float64)
        if fib.size < 10: fib = fat + (fat.std()+1e-3)*2
        if fat.size < 10: fat = fib - (fib.std()+1e-3)*2
        if fat.mean() <= fib.mean():
            mu1,sig1,w1 = fat.mean(),fat.std(),fat.size/interior.size
            mu2,sig2,w2 = fib.mean(),fib.std(),fib.size/interior.size
        else:
            mu1,sig1,w1 = fib.mean(),fib.std(),fib.size/interior.size
            mu2,sig2,w2 = fat.mean(),fat.std(),fat.size/interior.size
    def pdf(x):
        g1=w1*(1/(sig1*np.sqrt(2*np.pi)))*np.exp(-0.5*((x-mu1)/sig1)**2)
        g2=w2*(1/(sig2*np.sqrt(2*np.pi)))*np.exp(-0.5*((x-mu2)/sig2)**2)
        return g1+g2
    lo,hi = float(interior.min()), float(interior.max())
    gap = (mu2-sig2) - (mu1+sig1)
    delta = abs(gap)/2.0
    sep = gap > delta
    m_g=lo
    M_g=(mu1+sig1) if sep else (mu2-sig2-delta)
    mu_g=_expected(pdf,lo,hi,lambda x:x<M_g)
    m_sig_g=(mu1-sig1) if sep else (2*mu_g-M_g)
    m_f=mu2-sig2; M_f=hi
    mu_f=_expected(pdf,lo,hi,lambda x:x>m_f)
    m_sig_f=mu2+sig2
    bp = [m_g,m_sig_g,mu_g,M_g,m_f,mu_f,m_sig_f,M_f]
    bp = list(np.maximum.accumulate(bp))   # keep monotonic non-decreasing
    return bp, sep

def assign_subtissues(vol, label, x_break):
    """7 intensity bands (fat low/med/high, transition, fibro low/med/high) from precomputed
    breakpoints. Returns int array: 0=not interior, 1..7 bands. FREQUENCY-INDEPENDENT."""
    xb = np.asarray(x_break, float)
    breast = (label==FAT)|(label==FIBRO)
    out = np.zeros(vol.shape, np.int16)
    band = np.digitize(vol[breast], xb[1:-1])   # 6 inner edges -> 7 bins 0..6
    out[breast] = band + 1
    return out

# =====================================================================
# Cached per-source preparation: do the heavy work ONCE per breast.
# =====================================================================
def prepare_breaks(vol, label, sources=("seg","gmm")):
    """Compute breakpoints + sub-tissue bands once per source. Returns:
       { source: {'x_break': [...], 'sep': bool, 'subt': int-array} }."""
    out = {}
    for s in sources:
        x_break, sep = zastrow_breakpoints(vol, label, source=s)
        out[s] = {"x_break": x_break, "sep": sep,
                  "subt": assign_subtissues(vol, label, x_break)}
    return out

# =====================================================================
# "ours" conversion — frequency-dependent ONLY (fast; uses cached breakpoints)
# =====================================================================
def convert_ours(vol, label, x_break, freq_ghz):
    """Zastrow piecewise-linear over the 8 breakpoints. Fat/fibro interior only."""
    f = freq_ghz*GHZ
    y_er = [_cole_eval(f, COLE_CURVES[c])[0] for c in CURVE_ORDER]
    y_ei = [_cole_eval(f, COLE_CURVES[c])[1] for c in CURVE_ORDER]
    er = np.zeros_like(vol, np.float32); ei = np.zeros_like(vol, np.float32)
    breast = (label==FAT)|(label==FIBRO)
    I = vol[breast]
    er[breast] = np.interp(I, x_break, y_er)
    ei[breast] = np.interp(I, x_break, y_ei)
    return er, ei

# =====================================================================
# tumor / skin / muscle (same for both versions)
# =====================================================================
def assign_tumor_percentile(er, ei, label, vol, freq_ghz):
    """Map TUMOR voxels by intensity between malignant 25th/75th-percentile Debye curves."""
    m = label==TUMOR
    if m.sum()==0: return er, ei
    f=freq_ghz*GHZ
    er_lo,ei_lo = _debye_eval2(f, MALIGNANT_DEBYE["p25"])
    er_hi,ei_hi = _debye_eval2(f, MALIGNANT_DEBYE["p75"])
    I = vol[m]
    rank = (I-I.min())/(np.ptp(I)+1e-8) if I.size else np.zeros(m.sum())
    er[m] = er_lo + rank*(er_hi-er_lo)
    ei[m] = ei_lo + rank*(ei_hi-ei_lo)
    return er, ei

def segment_skin(label, thickness_vox=2):
    from scipy.ndimage import binary_erosion, binary_dilation
    breast = (label==FAT)|(label==FIBRO)|(label==TUMOR)
    if breast.sum()==0: return label.copy()
    eroded = binary_erosion(breast, iterations=thickness_vox)
    shell = breast & (~eroded)
    # don't put skin on the muscle-facing side (no skin between breast and chest wall)
    near_muscle = binary_dilation(label==MUSCLE, iterations=thickness_vox+1)
    shell = shell & (~near_muscle)
    out = label.copy(); out[shell] = SKIN_LABEL
    return out

def assign_skin(er, ei, label, freq_ghz, vol=None, heterogeneity=0.05, rng=None):
    m = label==SKIN_LABEL
    if m.sum()==0: return er, ei
    er0, ei0 = _cole4_eval(freq_ghz*GHZ, SKIN_GABRIEL)
    if rng is None: rng = np.random.RandomState(0)
    if vol is not None:
        I = vol[m]; rank = (I-I.min())/(np.ptp(I)+1e-8) if I.size else np.zeros(m.sum())
        scale = 1 - heterogeneity + 2*heterogeneity*rank
    else:
        scale = 1 + heterogeneity*(2*rng.rand(m.sum())-1)
    er[m] = er0*scale; ei[m] = ei0*scale
    return er, ei

def assign_muscle(er, ei, label, freq_ghz):
    m = label==MUSCLE
    if m.sum()==0: return er, ei
    er0, ei0 = _cole4_eval(freq_ghz*GHZ, MUSCLE_GABRIEL)
    er[m]=er0; ei[m]=ei0
    return er, ei
