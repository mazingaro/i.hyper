#!/usr/bin/env python3

import os, re, h5py, grass.script as gs
from grass.pygrass.modules import Module
from statistics import mean
from contextlib import contextmanager

COMPOSITES = {
    "RGB": [660, 572, 478],
    "CIR": [848, 660, 572],
    "SWIR-agriculture": [848, 1653, 660],
    "SWIR-geology": [2200, 848, 572],
}

@contextmanager
def suppress_stderr():
    import sys
    fd, old = sys.stderr.fileno(), os.dup(sys.stderr.fileno())
    try:
        with open(os.devnull, "w") as null:
            os.dup2(null.fileno(), fd)
        yield
    finally:
        os.dup2(old, fd); os.close(old)

def _find_hco_swath(h5):
    # prefer PRS_*_HCO (co-registered hypo cube)
    for name in h5.get("/HDFEOS/SWATHS", {}):
        if re.search(r"_HCO$", name):
            return f"/HDFEOS/SWATHS/{name}"
    # fall back to first swath with VNIR_Cube
    for name in h5.get("/HDFEOS/SWATHS", {}):
        base = f"/HDFEOS/SWATHS/{name}/Data Fields"
        if base in h5 and "VNIR_Cube" in h5[base]:
            return f"/HDFEOS/SWATHS/{name}"
    gs.fatal("No PRISMA swath with VNIR_Cube found.")

def _get_attr_anywhere(group, key, default=None):
    # look on swath, Data Fields, and dataset local attrs
    if key in group.attrs: return group.attrs[key]
    df = group.get("Data Fields")
    if isinstance(df, h5py.Group) and key in df.attrs: return df.attrs[key]
    for dname in ("VNIR_Cube","SWIR_Cube"):
        d = df.get(dname) if isinstance(df, h5py.Group) else None
        if isinstance(d, h5py.Dataset) and key in d.attrs:
            return d.attrs[key]
    return default

def _read_prisma_meta(he5_path):
    with h5py.File(he5_path, "r") as f:
        swath = _find_hco_swath(f)
        g = f[swath]
        df = g["Data Fields"]
        # Wavelengths / FWHM (nm)
        wl_v = _get_attr_anywhere(g, "List_Cw_Vnir")
        wl_s = _get_attr_anywhere(g, "List_Cw_Swir")
        fw_v = _get_attr_anywhere(g, "List_Fwhm_Vnir")
        fw_s = _get_attr_anywhere(g, "List_Fwhm_Swir")
        # Scale / offset (radiance = DN/Scale − Offset)
        sf_v = float(_get_attr_anywhere(g, "ScaleFactor_Vnir", 100.0))
        of_v = float(_get_attr_anywhere(g, "Offset_Vnir", 0.0))
        sf_s = float(_get_attr_anywhere(g, "ScaleFactor_Swir", 100.0))
        of_s = float(_get_attr_anywhere(g, "Offset_Swir", 0.0))
        # Sizes
        shape_v = df["VNIR_Cube"].shape  # (bands, lines, samples)
        shape_s = df["SWIR_Cube"].shape
    wl_v = list(map(float, wl_v)) if wl_v is not None else []
    wl_s = list(map(float, wl_s)) if wl_s is not None else []
    fw_v = list(map(float, fw_v)) if fw_v is not None else []
    fw_s = list(map(float, fw_s)) if fw_s is not None else []

    return {
        "swath": swath,
        "vnir_path": f'HDF5:"{he5_path}":{swath}/Data Fields/VNIR_Cube',
        "swir_path": f'HDF5:"{he5_path}":{swath}/Data Fields/SWIR_Cube',
        "wl_vnir": wl_v, "wl_swir": wl_s,
        "fwhm_vnir": fw_v, "fwhm_swir": fw_s,
        "scale_vnir": sf_v, "off_vnir": of_v,
        "scale_swir": sf_s, "off_swir": of_s,
        "shape_vnir": shape_v, "shape_swir": shape_s,
    }

def _nearest_idx(wl_target, wl_list):
    return min(range(len(wl_list)), key=lambda i: abs(wl_list[i] - wl_target)) if wl_list else None

def _stack_to_r3(band_maps, out3d):
    # Do NOT touch region; just stack as-is
    Module("r.to.rast3", input=band_maps, output=out3d, quiet=True, overwrite=True)

def import_prisma(folder, output, composites=None, custom_wavelengths=None, strength_val=96, keep_all_null=False):
    # find the only *.he5 in folder
    he5 = next((os.path.join(folder, x) for x in os.listdir(folder) if x.lower().endswith(".he5")), None)
    if not he5:
        gs.fatal("No .he5 file found in input directory.")

    meta = _read_prisma_meta(he5)

    # Build band list: VNIR then SWIR (preserve natural spectral ordering)
    band_maps = []
    wavelengths = []

    # helpers for exporting a single spectral band via GDAL subdataset + scaling to float
    def _export_band(subdataset, band_idx, name_prefix, scale, offset):
        raw = f"{name_prefix}_b{band_idx:03d}"
        with suppress_stderr():
            Module("r.external", input=subdataset, output=raw, band=band_idx, flags="o", quiet=True, overwrite=True)
        fout = f"{raw}_f"
        # FCELL scaling: float(DN/scale - offset)
        Module("r.mapcalc", expression=f"{fout}=float({raw}/{scale} - {offset})", quiet=True, overwrite=True)
        Module("g.remove", type="raster", name=raw, flags="f", quiet=True)
        return fout

    # VNIR bands
    n_v = meta["shape_vnir"][0]
    for b in range(1, n_v + 1):
        fout = _export_band(meta["vnir_path"], b, output, meta["scale_vnir"], meta["off_vnir"])
        band_maps.append(fout)
        if b-1 < len(meta["wl_vnir"]): wavelengths.append(meta["wl_vnir"][b-1])

    # SWIR bands
    n_s = meta["shape_swir"][0]
    for b in range(1, n_s + 1):
        fout = _export_band(meta["swir_path"], b, output, meta["scale_swir"], meta["off_swir"])
        band_maps.append(fout)
        if b-1 < len(meta["wl_swir"]): wavelengths.append(meta["wl_swir"][b-1])

    # Optional: drop all-NULL bands unless user asked to keep them
    if not keep_all_null:
        keep = []
        for m in band_maps:
            info = gs.parse_command("r.info", flags="r", map=m)
            if not (info.get("min") is None and info.get("max") is None):
                keep.append(m)
        band_maps = keep
        if not band_maps:
            gs.fatal("All bands are NULL after scaling.")

    # Composites (no region changes)
    if composites:
        def pick_maps(wls):
            idxs = []
            for wl in wls:
                i = _nearest_idx(wl, wavelengths)
                if i is None: gs.fatal("Wavelength list missing.")
                idxs.append(i)
            return [band_maps[i] for i in idxs]
        for comp in composites:
            if comp not in COMPOSITES: continue
            r,g,b = pick_maps(COMPOSITES[comp])
            if comp.upper() == "RGB":
                Module("i.colors.enhance", red=r, green=g, blue=b, strength=str(strength_val), flags="p", quiet=True)
            else:
                Module("i.colors.enhance", red=r, green=g, blue=b, strength=str(strength_val), quiet=True)
            outname = f"{output}_{comp.lower().replace('-','_')}"
            Module("r.composite", red=r, green=g, blue=b, output=outname, quiet=True, overwrite=True)
            gs.info(f"Generated composite raster: {outname}")

    # Spectral axis spacing for r3 metadata
    if len(wavelengths) > 1:
        diffs = [wavelengths[i+1]-wavelengths[i] for i in range(len(wavelengths)-1)]
        tbres_nm = float(f"{max(1e-6, mean(diffs)):.6f}")
    else:
        tbres_nm = 1.0
    bottom_nm = wavelengths[0] if wavelengths else 0.0
    top_nm = bottom_nm + tbres_nm * (len(wavelengths)-1)

    # Stack -> r3, then annotate
    _stack_to_r3(band_maps, output)

    desc = [ "Hyperspectral Metadata:", f"Valid Bands: {len(wavelengths)}" ]
    # write per-band goodies on the 2D maps first (title units, wavelength/fwhm)
    for i, m in enumerate(band_maps, 1):
        wl = wavelengths[i-1] if i-1 < len(wavelengths) else None
        fwhm = None
        if i-1 < len(meta["fwhm_vnir"]): fwhm = meta["fwhm_vnir"][i-1]
        else:
            j = i-1 - len(meta["wl_vnir"])
            if 0 <= j < len(meta["fwhm_swir"]): fwhm = meta["fwhm_swir"][j]
        desc.append(f"Band {i}: {wl} nm, FWHM: {fwhm} nm")
        if wl is not None:
            Module("r.support", map=m, title=f"Band {i}", units="nm",
                   source1=f"Wavelength: {wl} nm",
                   source2=f"FWHM: {fwhm} nm" if fwhm is not None else "",
                   description="PRISMA band", quiet=True)

    Module("r3.support", map=output,
           title="PRISMA Hyperspectral Data",
           description="\n".join(desc),
           vunit="nanometers", quiet=True)

    # Clean 2D band rasters after stacking
    Module("g.remove", type="raster", name=band_maps, flags="f", quiet=True)

def run_import(options, flags):
    import_prisma(
        folder=options["input"],
        output=options["output"],
        composites=[c.strip() for c in options["composites"].split(",")] if options.get("composites") else None,
        custom_wavelengths=([float(x.strip()) for x in options["composites_custom"].split(",")]
                            if options.get("composites_custom") else None),
        strength_val=int(options.get("strength") or 96),
        keep_all_null=("n" in flags),
    )
