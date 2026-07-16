"""Comparison plots for the two 'ours' versions (segmentation vs gmm)."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

VIEW_AXIS = 0
VERSION_NAMES = ["ours_segmentation", "ours_gmm"]

def _best_slice(label, axis=VIEW_AXIS):
    has = [i for i in range(label.shape[axis]) if (np.take(label,i,axis)>0).any()]
    if not has: return axis, label.shape[axis]//2
    return axis, int((has[0]+has[-1])//2)

def two_version_maps(breasts_data, freq, out_path, view_axis=VIEW_AXIS):
    """Both breasts x 2 versions, eps' and eps'' at each breast's tumor (or middle) slice.
    breasts_data: list of {name, label, results{version:(er,ei)}}."""
    nb = len(breasts_data); nrow = nb*2; ncol = len(VERSION_NAMES)
    fig, ax = plt.subplots(nrow, ncol, figsize=(4.2*ncol, 4*nrow))
    if nrow == 1: ax = ax[None,:]
    if ncol == 1: ax = ax[:,None]
    for bi, bd in enumerate(breasts_data):
        label = bd["label"]; res = bd["results"]; breast = label>0
        tcounts=[(np.take(label,i,view_axis)==3).sum() for i in range(label.shape[view_axis])]
        if max(tcounts)>0: z=int(np.argmax(tcounts)); tag=f"tumor slice {z}"
        else: _,z=_best_slice(label); tag=f"slice {z}"
        allr=np.concatenate([res[v][0][breast] for v in VERSION_NAMES])
        alli=np.concatenate([res[v][1][breast] for v in VERSION_NAMES])
        vmaxr=np.nanpercentile(allr,99); vmaxi=np.nanpercentile(alli,99)
        for ci, v in enumerate(VERSION_NAMES):
            er,ei = res[v]
            erm=np.where(breast,er,np.nan); eim=np.where(breast,ei,np.nan)
            r0=bi*2
            im0=ax[r0,ci].imshow(np.take(erm,z,view_axis).T,cmap='jet',origin='lower',vmin=0,vmax=vmaxr)
            ax[r0,ci].set_title(f"{bd['name']} {v}\nε'  ({tag})",fontsize=10); ax[r0,ci].axis('off')
            plt.colorbar(im0,ax=ax[r0,ci],fraction=0.046)
            im1=ax[r0+1,ci].imshow(np.take(eim,z,view_axis).T,cmap='hot',origin='lower',vmin=0,vmax=vmaxi)
            ax[r0+1,ci].set_title(f"{bd['name']} {v}\nε''",fontsize=10); ax[r0+1,ci].axis('off')
            plt.colorbar(im1,ax=ax[r0+1,ci],fraction=0.046)
    plt.suptitle(f"Permittivity & conductivity — ours seg vs gmm, both breasts @ {freq} GHz",
                 fontweight='bold',fontsize=14)
    plt.tight_layout(); plt.savefig(out_path,dpi=120,bbox_inches='tight'); plt.close()

def _kde_curve(ax, vals, xgrid, color, label, max_n=40000):
    from scipy.stats import gaussian_kde
    vals = vals[np.isfinite(vals)]
    if vals.size < 5: return
    if np.ptp(vals) < 1e-6:
        ax.axvline(vals[0], color=color, lw=2, label=label); return
    if vals.size > max_n:
        vals = np.random.default_rng(0).choice(vals, size=max_n, replace=False)
    try:
        kde = gaussian_kde(vals)
        ax.plot(xgrid, kde(xgrid), color=color, lw=2, label=label)
        ax.fill_between(xgrid, kde(xgrid), color=color, alpha=0.12)
    except Exception:
        ax.hist(vals, bins=60, density=True, histtype='step', lw=2, color=color, label=label)

def two_version_distributions(breasts_data, freq, out_path, part="real"):
    """2 panels (one per version). KDE curves for grouped tissues:
    fat (bands 1-3), fibroglandular (bands 5-7), transition (band 4), tumor."""
    comp = 0 if part=="real" else 1
    xmax = 60 if part=="real" else 28
    xlab = "ε'" if part=="real" else "ε''"
    xgrid = np.linspace(0, xmax, 400)
    groups = [
        ("fat",            "tab:blue",  lambda lab,subt: np.isin(subt,[1,2,3])),
        ("fibroglandular", "tab:red",   lambda lab,subt: np.isin(subt,[5,6,7])),
        ("transition",     "tab:olive", lambda lab,subt: subt==4),
        ("tumor",          "tab:green", lambda lab,subt: lab==3),
    ]
    fig, ax = plt.subplots(1, len(VERSION_NAMES), figsize=(8*len(VERSION_NAMES), 6))
    if len(VERSION_NAMES)==1: ax=[ax]
    for vi, v in enumerate(VERSION_NAMES):
        a = ax[vi]
        src = "seg" if v.endswith("segmentation") else "gmm"
        for gname,gcol,gsel in groups:
            vals=[]
            for bd in breasts_data:
                er = bd["results"][v][comp]; lab = bd["label"]; subt = bd["results"][f"subt_{src}"]
                vals.append(er[gsel(lab,subt)])
            vals = np.concatenate(vals) if vals else np.array([])
            _kde_curve(a, vals, xgrid, gcol, f"{gname} (n={vals.size})")
        a.set_title(v, fontweight='bold'); a.set_xlabel(xlab); a.set_ylabel("density")
        a.set_xlim(0,xmax); a.legend(fontsize=9)
    plt.suptitle(f"{xlab} distributions (KDE) by tissue — per version @ {freq} GHz",
                 fontweight='bold', fontsize=14)
    plt.tight_layout(); plt.savefig(out_path,dpi=120,bbox_inches='tight'); plt.close()



def mri_vs_permittivity(vol_orig, vol_pre, er, out_png, title=""):
    import matplotlib.pyplot as plt
    import numpy as np
    z = int(np.argmax((er > 1.5).sum(axis=(1, 2))))   # richest axial slice
    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    axes[0].imshow(vol_orig[z, :, :], cmap='gray', origin='lower')
    axes[0].set_title(f"raw MRI  z={z}"); axes[0].axis('off')
    axes[1].imshow(vol_pre[z, :, :], cmap='gray', origin='lower')
    axes[1].set_title("preprocessed (N4+median)"); axes[1].axis('off')
    im = axes[2].imshow(er[z, :, :], cmap='jet', vmin=0, vmax=60, origin='lower')
    axes[2].set_title("ε′"); axes[2].axis('off')
    fig.colorbar(im, ax=axes[2], fraction=0.046)
    if title: fig.suptitle(title, fontsize=12)
    plt.tight_layout(); plt.savefig(out_png, dpi=130, bbox_inches="tight"); plt.close()
