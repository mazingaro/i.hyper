#!/usr/bin/env python3
# PRISMA importer backend for i.hyper.import

import os
import re
import h5py
import grass.script as gs
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
        os.dup2(old, fd)
        os.close(old)

def _find_hco_swath(h5: h5py.File) -> str:
    # Prefer *_HCO swath (co-registered hyperspectral cube)
    swaths = h5.get("/HDFEOS/SWATHS")
    if not isinstance(swaths, h5py.Group):
        gs.fatal("PRISMA HE5: /HDFEOS/SWATHS group not found.")
    for name in swaths:
        if re.search(r"_HCO$", name):
            return f"/HDFEOS/SWATHS/{name}"
    # Fallback: first swath that has VNIR_Cube under "Data Fields"
    for name in swaths:
        base = f"/HDFEOS/SWATHS/{name}/Data Fields"
        if base in h5 and "VNIR_Cube" in h5[base]:
            return f"/HDFEOS/SWATHS/{name}"
    gs.fatal("No PRISMA swath with VNIR_Cube found.")

def _get_attr_anywhere(swath_group: h5py.Group, key: str, default=None):
    # Search on swath, "Data Fields", and the VNIR/SWIR datasets
    if key in swath_group.attrs:
        return swath_group.attrs[key]
    df = swath_group.get("Data Fields")
    if isinstance(df, h5py.Group) and key in df.attrs:
        return df.attrs[key]
    for dname in ("VNIR_Cube", "SWIR_Cube"):
        d = df.get(dname) if isinstance(df, h5py.Group) else None
        if isinstance(d, h5py.Dataset) and key in d.attrs:
            return d.attrs[key]
    return default

def _read_prisma_meta(he5_path: str):
    with h5py.File(he5_path, "r") as f:
        swath = _find_hco_swath(f)
        g = f[swath]

        df = g.get("Data Fields")
        if not isinstance(df, h5py.Group):
            gs.fatal(f"'{swath}/Data Fields' not found (note the space).")

        # Shapes are (lines, bands, samples) per your listing
        if "VNIR_Cube" not in df or "SWIR_Cube" not in df:
            gs.fatal("VNIR_Cube or SWIR_Cube dataset missing in Data Fields.")
        shape_v = df["VNIR_Cube"].shape
        shape_s = df["SWIR_Cube"].shape
        n_vnir = shape_v[1]  # bands are at index 1
        n_swir = shape_s[1]  # bands are at index 1

        # Wavelengths / FWHM (nm) from attributes
        wl_v = _get_attr_anywhere(g, "List_Cw_Vnir")
        wl_s = _get_attr_anywhere(g, "List_Cw_Swir")
        fw_v = _get_attr_anywhere(g, "List_Fwhm_Vnir")
        fw_s = _get_attr_anywhere(g, "List_Fwhm_Swir")

        wl_v = [float(x) for x in wl_v] if wl_v is not None else []
        wl_s = [float(x) for x in wl_s] if wl_s is not None else []
        fw_v = [float(x) for x in fw_v] if fw_v is not None else []
        fw_s = [float(x) for x in fw_s] if fw_s is not None else []

        # Scale/offset (radiance or reflectance): value = DN/Scale - Offset
        # If absent, fall back to 10000 / 0 and warn.
        def _num(x, dflt):
            try:
                return float(x)
            except Exception:
                return dflt

        sv = _get_attr_anywhere(g, "ScaleFactor_Vnir")
        ov = _get_attr_anywhere(g, "Offset_Vnir")
        ss = _get_attr_anywhere(g, "ScaleFactor_Swir")
        os_ = _get_attr_anywhere(g, "Offset_Swir")

        scale_v = _num(sv, 10000.0)
        off_v   = _num(ov, 0.0)
        scale_s = _num(ss, 10000.0)
        off_s   = _num(os_, 0.0)

        if sv is None or ss is None:
            gs.message("PRISMA: ScaleFactor attribute missing, using 10000.0 as default.")
        if ov is None or os_ is None:
            gs.message("PRISMA: Offset attribute missing, using 0.0 as default.")

    # CORRECT GDAL subdataset path format
    # GDAL expects: HDF5:"filename"://path/to/dataset
    # Remove leading slash and use proper HDF5 path structure
    swath_path = swath.lstrip('/')
    vnir_path = f'{swath_path}/Data Fields/VNIR_Cube'
    swir_path = f'{swath_path}/Data Fields/SWIR_Cube'
    
    return {
        "swath": swath,
        "vnir_sds": f'HDF5:"{he5_path}"://{vnir_path}',
        "swir_sds": f'HDF5:"{he5_path}"://{swir_path}',
        "shape_vnir": shape_v, "shape_swir": shape_s,
        "n_vnir": n_vnir, "n_swir": n_swir,
        "wl_vnir": wl_v, "wl_swir": wl_s,
        "fwhm_vnir": fw_v, "fwhm_swir": fw_s,
        "scale_vnir": scale_v, "off_vnir": off_v,
        "scale_swir": scale_s, "off_swir": off_s,
    }

def _nearest_idx(wl_target, wl_list):
    return min(range(len(wl_list)), key=lambda i: abs(wl_list[i] - wl_target)) if wl_list else None

def _export_band_to_fcell(subdataset: str, band_idx: int, base: str, scale: float, offset: float) -> str:
    """Link one band via r.external, then scale to FCELL with r.mapcalc float()."""
    raw = f"{base}_b{band_idx:03d}"
    
    # First test if GDAL can read the subdataset
    try:
        with suppress_stderr():
            # Test with gdalinfo first to see if the subdataset is accessible
            test_cmd = f'gdalinfo "{subdataset}"'
            gs.run_command('bash', '-c', test_cmd, quiet=True)
    except:
        gs.fatal(f"Cannot access subdataset: {subdataset}")
    
    with suppress_stderr():
        Module("r.external", input=subdataset, output=raw, band=band_idx, flags="o", quiet=True, overwrite=True)
    fout = f"{raw}_f"
    # FCELL expression
    Module("r.mapcalc", expression=f"{fout}=float({raw}/{scale} - {offset})", quiet=True, overwrite=True)
    # Clean the GDAL link
    Module("g.remove", type="raster", name=raw, flags="f", quiet=True)
    return fout

def _stack_to_r3(band_maps, out3d):
    # Do NOT touch user's region settings
    Module("r.to.rast3", input=band_maps, output=out3d, quiet=True, overwrite=True)

def import_prisma(file_path, output, composites=None, custom_wavelengths=None, strength_val=96, keep_all_null=False):
    if not os.path.isfile(file_path):
        gs.fatal(f"Input is not a file: {file_path}")

    meta = _read_prisma_meta(file_path)
    gs.message(f"VNIR subdataset: {meta['vnir_sds']}")
    gs.message(f"SWIR subdataset: {meta['swir_sds']}")

    # Build VNIR then SWIR band list
    band_maps = []
    wavelengths = []

    # VNIR
    gs.message(f"Importing {meta['n_vnir']} VNIR bands...")
    for b in range(1, meta["n_vnir"] + 1):
        try:
            fout = _export_band_to_fcell(meta["vnir_sds"], b, output, meta["scale_vnir"], meta["off_vnir"])
            band_maps.append(fout)
            if b - 1 < len(meta["wl_vnir"]):
                wavelengths.append(meta["wl_vnir"][b - 1])
        except Exception as e:
            gs.warning(f"Failed to import VNIR band {b}: {e}")
            continue

    # SWIR
    gs.message(f"Importing {meta['n_swir']} SWIR bands...")
    for b in range(1, meta["n_swir"] + 1):
        try:
            fout = _export_band_to_fcell(meta["swir_sds"], b, output, meta["scale_swir"], meta["off_swir"])
            band_maps.append(fout)
            if b - 1 < len(meta["wl_swir"]):
                wavelengths.append(meta["wl_swir"][b - 1])
        except Exception as e:
            gs.warning(f"Failed to import SWIR band {b}: {e}")
            continue

    if not band_maps:
        gs.fatal("No bands were successfully imported.")

    # Optionally drop all-NULL bands
    if not keep_all_null:
        keep_maps, keep_wls = [], []
        for i, m in enumerate(band_maps):
            info = gs.parse_command("r.info", flags="r", map=m)
            if not (info.get("min") is None and info.get("max") is None):
                keep_maps.append(m)
                if i < len(wavelengths):
                    keep_wls.append(wavelengths[i])
        band_maps, wavelengths = keep_maps, keep_wls
        if not band_maps:
            gs.fatal("All bands are NULL after scaling.")

    # Composites (no region changes)
    if composites:
        def pick_maps(wls):
            idxs = []
            for wl in wls:
                i = _nearest_idx(wl, wavelengths)
                if i is None:
                    gs.fatal("Wavelength list missing.")
                idxs.append(i)
            return [band_maps[i] for i in idxs]
        
        for comp in composites:
            if comp not in COMPOSITES:
                continue
            r, g, b = pick_maps(COMPOSITES[comp])
            if comp.upper() == "RGB":
                Module("i.colors.enhance", red=r, green=g, blue=b, strength=str(strength_val), flags="p", quiet=True)
            else:
                Module("i.colors.enhance", red=r, green=g, blue=b, strength=str(strength_val), quiet=True)
            outname = f"{output}_{comp.lower().replace('-', '_')}"
            Module("r.composite", red=r, green=g, blue=b, output=outname, quiet=True, overwrite=True)
            gs.info(f"Generated composite raster: {outname}")

    # Stack -> 3D
    _stack_to_r3(band_maps, output)

    # Per-band metadata on 2D rasters and volume description
    desc = ["Hyperspectral Metadata:", f"Valid Bands: {len(wavelengths)}"]

    # Prepare FWHM lookup aligned with our VNIR+SWIR concatenation
    fwhm_all = (meta["fwhm_vnir"] or []) + (meta["fwhm_swir"] or [])
    for i, m in enumerate(band_maps, 1):
        wl = wavelengths[i - 1] if i - 1 < len(wavelengths) else None
        fwhm = fwhm_all[i - 1] if i - 1 < len(fwhm_all) else None
        desc.append(f"Band {i}: {wl} nm, FWHM: {fwhm} nm")
        title = f"Band {i}"
        s1 = f"Wavelength: {wl} nm" if wl is not None else ""
        s2 = f"FWHM: {fwhm} nm" if fwhm is not None else ""
        Module("r.support", map=m, title=title, units="nm", source1=s1, source2=s2,
               description="PRISMA band", quiet=True)

    Module("r3.support", map=output,
           title="PRISMA Hyperspectral Data",
           description="\n".join(desc),
           vunit="nanometers", quiet=True)

    # Clean 2D FCELL rasters after stacking
    Module("g.remove", type="raster", name=band_maps, flags="f", quiet=True)

def run_import(options, flags):
    import_prisma(
        file_path=options["input"],
        output=options["output"],
        composites=[c.strip() for c in options["composites"].split(",")] if options.get("composites") else None,
        custom_wavelengths=([float(x.strip()) for x in options["composites_custom"].split(",")]
                            if options.get("composites_custom") else None),
        strength_val=int(options.get("strength") or 96),
        keep_all_null=("n" in flags),
    )