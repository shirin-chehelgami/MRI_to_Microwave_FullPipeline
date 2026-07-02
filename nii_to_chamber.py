#!/usr/bin/env python3
"""
nii_to_chamber.py
Breast (label + eps'/eps'' NIfTI) -> CHAMBER voxel file (synthetic_blue_ball.mat format).

AXIS ORDER = "z, x, y" (as specified):
   axis 0 = chest-depth   (the cup axis; blue line/rim at index D0, mound hangs to lower indices)
   axis 1 = lateral
   axis 2 = sup-inf
All tissue kept (fat,fibro,tumor,muscle,skin). Mound below the rim (into cup),
chest wall above the rim. Centred on axis1 = axis2 = 0.

Usage:
    python nii_to_chamber.py label.nii.gz real.nii.gz imag.nii.gz out.mat [out.png]
"""
import sys, numpy as np, nibabel as nib
from scipy.ndimage import label as cc_label, binary_fill_holes, zoom, sum as ndsum
from scipy.io import savemat

def _largest_cc(m):
    lab,n=cc_label(m)
    return m if n<=1 else (lab==1+int(np.argmax(ndsum(m,lab,range(1,n+1)))))

def measure_valley(L, breast_labels=(1,2,3,5), muscle_label=4, tumor_label=3):
    B=np.isin(L,breast_labels); mus=(L==muscle_label)
    sc=[abs(np.where(mus.any(axis=tuple(y for y in range(3) if y!=a)))[0].mean()/L.shape[a]-0.5)
        if mus.any() else -1 for a in range(3)]
    cax=int(np.argmax(sc)); rest=[a for a in range(3) if a!=cax]
    def asym(a):
        m=B.sum(axis=tuple(x for x in range(3) if x!=a)).astype(float); nz=np.where(m>0)[0]
        m=m[nz.min():nz.max()+1]; return np.abs(m-m[::-1]).sum()/(m.sum()+1e-9)
    lax=rest[int(np.argmax([asym(a) for a in rest]))]; sax=[a for a in rest if a!=lax][0]
    per=(L==tumor_label).sum(axis=tuple(a for a in range(3) if a!=sax))
    if per.sum()<10: per=B.sum(axis=tuple(a for a in range(3) if a!=sax))
    si=int(np.argmax(per))
    sl2=np.take(B,si,axis=sax); order=[a for a in range(3) if a!=sax]
    sl=np.moveaxis(sl2,[order.index(lax),order.index(cax)],[0,1])
    sl=_largest_cc(binary_fill_holes(sl)); nLv,nCv=sl.shape
    cpres=np.where(sl.any(axis=0))[0]
    chest_high=mus.any() and (np.where(mus.any(axis=tuple(x for x in range(3) if x!=cax)))[0].mean()>L.shape[cax]/2)
    nip_c=cpres.min() if chest_high else cpres.max()
    chest_c=cpres.max() if chest_high else cpres.min()
    lpres=np.where(sl.any(axis=1))[0]; medial_low=lpres.min()<=(nLv-1-lpres.max())
    med=np.array([(np.where(sl[:,c])[0].min() if medial_low else np.where(sl[:,c])[0].max())
                  if sl[:,c].any() else np.nan for c in range(nCv)])
    extreme=np.nanmin(med) if medial_low else np.nanmax(med); rng=np.nanmax(med)-np.nanmin(med)
    thr=extreme+0.15*rng if medial_low else extreme-0.15*rng
    scan=range(nip_c,chest_c+(1 if chest_high else -1),1 if chest_high else -1)
    prev=None; vC=None
    for c in scan:
        if np.isnan(med[c]): continue
        hit=med[c]<=thr if medial_low else med[c]>=thr
        if hit and prev is not None: vC=prev; break
        prev=c
    if vC is None: vC=int(cpres[len(cpres)//2])
    vL=int(med[vC]); ys=np.where(sl[:,vC])[0]; end=vL
    if medial_low:
        while end+1 in ys: end+=1
    else:
        while end-1 in ys: end-=1
    return dict(cax=cax,lax=lax,sax=sax,si=si,chest_high=bool(chest_high),vC=int(vC),vL=int(vL),end=int(end))

def make_chamber(label_path, real_path, imag_path, out_mat,
                 diam_cm=22.0, depth_cm=15.0, vox_mm=1.0, tissues=(1,2,3,4,5)):
    lab=nib.load(label_path); sp=np.array(lab.header.get_zooms()[:3],float)
    L=lab.get_fdata().astype(np.int16)
    RE=nib.load(real_path).get_fdata(); IM=nib.load(imag_path).get_fdata()
    zf=sp/vox_mm
    Lr=zoom(L,zf,order=0); REr=zoom(RE,zf,order=1); IMr=zoom(IM,zf,order=1)
    r=measure_valley(Lr,breast_labels=tuple(t for t in tissues if t!=4))
    cax,lax,sax=r["cax"],r["lax"],r["sax"]; vC=r["vC"]; vL=r["vL"]; end=r["end"]
    si=r["si"]; chest_high=r["chest_high"]; lat_mid=(vL+end)//2
    step=vox_mm/1000.0

    eps=np.where(np.isin(Lr,tissues), REr-1j*IMr, 1.0+0j)
    # eps[Lr==4] = 1.0+0j  # For without muscle test only
    # AXIS ORDER "z,x,y": (chest-depth, lateral, sup-inf) -> (axis0, axis1, axis2)
    epsC=np.moveaxis(eps,[cax,lax,sax],[0,1,2])
    tisC=np.moveaxis(np.isin(Lr,tissues),[cax,lax,sax],[0,1,2])
    nC,nL,nS=epsC.shape                                    # chest-depth, lateral, sup-inf

    NL=int(round(diam_cm*10/vox_mm)); NS=NL                 # lateral(axis1), supinf(axis2) grid
    cup_below=int(round(depth_cm*10/vox_mm))
    Xs=np.where(tisC.any(axis=(1,2)))[0]                    # chest-depth indices with tissue
    below=(vC-Xs.min()) if chest_high else (Xs.max()-vC)   # mound depth (voxels)
    above=(Xs.max()-vC) if chest_high else (vC-Xs.min())   # chest-wall height (voxels)
    D0=max(cup_below,below)                                 # rim index on chest-depth axis (axis0)
    ND=D0+above+1
    data=np.ones((ND,NL,NS),dtype=np.complex128)           # air ; axis0=chest-depth

    # chest-depth index X -> chamber axis0 index: valley(vC)->D0 ; mound below ; chest wall above
    zof=lambda X: D0 + ((X-vC) if chest_high else (vC-X))
    offL=NL//2-lat_mid; offS=NS//2-si
    for X in range(nC):
        zi=zof(X)
        if not(0<=zi<ND): continue
        src=epsC[X]                                         # (lateral, sup-inf) plane
        l0=max(0,offL); l1=min(NL,nL+offL); s0=max(0,offS); s1=min(NS,nS+offS)
        seg=src[l0-offL:l1-offL, s0-offS:s1-offS]; m=(seg!=(1.0+0j))
        data[zi,l0:l1,s0:s1][m]=seg[m]

    # origin: blue line (axis0=D0) at 0 ; lateral & supinf centred
    origin=np.array([[-(NS//2)*step, -(NL//2)*step, -D0*step]])  # reader order: [x(axis2), y(axis1), z(axis0=depth)]
    steps =np.array([[step,step,step]])
    savemat(out_mat,{"data":data,"origin":origin,"steps":steps})
    return data,origin[0],step,D0,NL,NS,ND

def plot_chamber(data,origin,step,D0,NL,NS,out_png):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe
    cy=NL//2; opening=D0
    sl=data.real[:,:,NS//2]; breast=sl>1.5                  # (chest-depth, lateral) profile
    xb=np.where(breast.any(axis=1))[0]; nip=int(xb.min())
    depth_cm=(opening-nip)*step*100
    st=[pe.withStroke(linewidth=3,foreground='black')]
    def T(ax,x,y,s,c='white',**k): ax.text(x,y,s,color=c,fontweight='bold',path_effects=st,**k)
    fig,ax=plt.subplots(figsize=(9,8.5))
    ax.imshow(sl,origin="lower",cmap="jet",vmin=0,vmax=50,aspect=1)
    ax.axhline(opening,color='white',ls='--',lw=2,path_effects=st)
    T(ax,4,opening+3,"blue line = rim (axis0=0)",fontsize=10)
    T(ax,4,opening+12,"chest wall ABOVE",fontsize=9)
    T(ax,4,opening-12,"mound BELOW (cup)",fontsize=9,va='top')
    ax.set_ylabel("axis0 = chest-depth (%d=rim)"%opening); ax.set_xlabel("axis1 = lateral")
    ax.set_title('axis order z,x,y = (chest-depth, lateral, sup-inf)')
    plt.tight_layout(); plt.savefig(out_png,dpi=120,bbox_inches="tight"); plt.close()

if __name__=="__main__":
    a=sys.argv
    label,real,imag,out_mat=a[1],a[2],a[3],a[4]
    out_png=a[5] if len(a)>5 else out_mat.rsplit(".",1)[0]+".png"
    data,origin,step,D0,NL,NS,ND=make_chamber(label,real,imag,out_mat)
    plot_chamber(data,origin,step,D0,NL,NS,out_png)
    re=data.real
    print("saved:",out_mat,"and",out_png)
    print("shape (axis0=chest-depth, axis1=lateral, axis2=supinf):",data.shape)
    print("origin(m)",tuple(round(v,3) for v in origin))
    for ax,nm in [(0,"axis0 chest-depth"),(1,"axis1 lateral"),(2,"axis2 supinf")]:
        idx=np.where((re>1.5).any(axis=tuple(q for q in range(3) if q!=ax)))[0]
        print(f"  {nm}: {round(origin[ax]+idx.min()*step,3)} .. {round(origin[ax]+idx.max()*step,3)} m")
