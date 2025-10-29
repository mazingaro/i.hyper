#!/usr/bin/env python3
# Tanager → GRASS (3D + composites), EnMAP-style background handling
# - Entry: run_import() → import_tanager(...)
# - Writes float32 (SR or radiance) with NaNs outside footprint (NULL in GRASS)
# - Temp 2D bands for composites; cleans up afterwards

import os
import uuid
import numpy as np
import grass.script as gs
import grass.script.array as garray
from grass.pygrass.modules import Module

from tanager_importer import load_tanager_basic


# ---- Inlined importer code (was a separate module) ----
#!/usr/bin/env python3
"""
Tanager BASIC reader (radiance or surface_reflectance)

- Reads (Band, Y, X) cube and transposes to (rows=Y, cols=X, bands).
- Extracts wavelengths/FWHM from dataset attributes.
- Applies outside-footprint mask using /HDFEOS/SWATHS/HYP/Data Fields/nodata_pixels (1 -> NaN).
- Exposes per-pixel lat/lon grids.
- Ensures ascending wavelength order (and reorders cube accordingly).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any
import numpy as np
import h5py

# ---- Inlined common helpers ----
def _require(cond, msg):
    if not cond:
        gs.fatal(msg)

def _find_nearest_band_1based(target_nm, wavelengths_nm):
    wl = np.asarray(wavelengths_nm, dtype=np.float32)
    return int(np.argmin(np.abs(wl - float(target_nm)))) + 1  # 1-based

def _temp_name(prefix):
    return f"{prefix}_{uuid.uuid4().hex[:8]}"
# ---- End inlined helpers ----


# ---- HDF5 layout ----
HYP = "/HDFEOS/SWATHS/HYP"
DF  = f"{HYP}/Data Fields"
GF  = f"{HYP}/Geolocation Fields"

DS_ORDER = ("surface_reflectance", "toa_radiance")  # prefer SR if present
DS_WL_ATTR   = "wavelengths"
DS_FWHM_ATTR = "fwhm"

DS_NODATA = f"{DF}/nodata_pixels"
DS_LAT    = f"{GF}/Latitude"
DS_LON    = f"{GF}/Longitude"

@dataclass
class TanagerProduct:
    path: str
    data: np.ndarray              # (rows, cols, bands), float32
    wavelengths_nm: np.ndarray    # (bands,)
    fwhm_nm: np.ndarray           # (bands,)
    lat: Optional[np.ndarray]     # (rows, cols)
    lon: Optional[np.ndarray]     # (rows, cols)
    attrs: Dict[str, Any]         # file/global attrs (optional use)

def _maybe(f: h5py.File, path: str):
    return f[path][()] if path in f else None

def load_tanager_basic(product_path: str) -> TanagerProduct:
    with h5py.File(product_path, "r") as f:
        # keep top-level attrs (optional)
        attrs: Dict[str, Any] = {}
        for k, v in f.attrs.items():
            try:
                attrs[k] = v.decode(errors="ignore") if isinstance(v, (bytes, bytearray)) else (
                    np.array(v).tolist() if isinstance(v, np.ndarray) else v
                )
            except Exception:
                attrs[k] = v

        # pick SR if available, else radiance
        dset = None
        for name in DS_ORDER:
            p = f"{DF}/{name}"
            if p in f:
                dset = f[p]
                break
        if dset is None:
            raise ValueError("Tanager: neither 'surface_reflectance' nor 'toa_radiance' found")

        arr_raw = dset[()]  # (Band, Y, X), float32 per spec
        if arr_raw.ndim != 3:
            raise ValueError(f"Unexpected Tanager dims {arr_raw.shape} (expected 3D)")

        # transpose to (rows, cols, bands)
        data = np.transpose(arr_raw, (1, 2, 0)).astype(np.float32, copy=False)

        # spectral meta
        wl   = np.array(dset.attrs[DS_WL_ATTR], dtype=np.float32)
        fwhm = np.array(dset.attrs.get(DS_FWHM_ATTR, np.full_like(wl, np.nan, dtype=np.float32)), dtype=np.float32)

        # ensure ascending wavelength order (and reorder cube & fwhm)
        order = np.argsort(wl.astype(np.float32))
        wl = wl[order]
        fwhm = fwhm[order]
        data = data[:, :, order]

        # geolocation
        lat = _maybe(f, DS_LAT)
        lon = _maybe(f, DS_LON)

        # ---- EnMAP-style exterior mask: nodata_pixels == 1 -> NaN
        nd = _maybe(f, DS_NODATA)
        if nd is not None:
            if nd.shape != data.shape[:2]:
                raise ValueError(f"nodata_pixels shape {nd.shape} != image plane {data.shape[:2]}")
            m = nd.astype(bool)
            if m.any():
                data[m, :] = np.nan

    return TanagerProduct(
        path=product_path,
        data=data,
        wavelengths_nm=wl,
        fwhm_nm=fwhm,
        lat=lat,
        lon=lon,
        attrs=attrs,
    )
# ---- End inlined importer code ----


COMPOSITES = {
    "RGB":              [660.0, 572.0, 478.0],
    "CIR":              [848.0, 660.0, 572.0],
    "SWIR-agriculture": [848.0, 1653.0, 660.0],
    "SWIR-geology":     [2200.0, 848.0, 572.0],
}
# -------------------------- helpers --------------------------

def _resolve_h5(path_like):
    if os.path.isdir(path_like):
        for n in os.listdir(path_like):
            if n.lower().endswith(".h5"):
                return os.path.join(path_like, n)
        gs.fatal("No .h5 file found in the provided folder.")
    return path_like



def _write_float_raster(name, data_2d_float32):
    arr = garray.array(dtype=np.float32)
    arr[:, :] = data_2d_float32
    arr.write(name, null="nan", overwrite=True)  # NaNs -> NULLs

# -------------------------- core --------------------------
def import_tanager(input_path, output_name, composites=None, custom_wavelengths=None, strength_val=96, import_null=False):
    h5 = _resolve_h5(input_path)
    prod = load_tanager_basic(h5)

    data = prod.data              # (N,E,B) float32, NaNs where nodata_pixels==1
    wl   = prod.wavelengths_nm
    fwhm = prod.fwhm_nm

    _require(data is not None and data.ndim == 3, "Tanager cube missing or invalid.")
    rows, cols, bands_total = data.shape

    # Determine transposed shape (E,N) from any band and set a basic rows/cols region
    first_band = data[:, :, 0].T
    rows_E, cols_N = first_band.shape

    gs.use_temp_region()
    Module("g.region", rows=rows_E, cols=cols_N, quiet=True)

    # Build list of composites (default RGB), safe lookup like PRISMA
    wanted = []
    if composites:
        comp_lookup = {k.upper(): (k, v) for k, v in COMPOSITES.items()}
        for comp in composites:
            compu = comp.strip().upper()
            if compu in comp_lookup:
                orig_name, vals = comp_lookup[compu]
                wanted.append((orig_name, vals))
            else:
                gs.warning(f"Unknown composite '{comp}' ignored.")
    else:
        wanted.append(("RGB", COMPOSITES["RGB"]))

    if custom_wavelengths:
        if len(custom_wavelengths) != 3:
            gs.fatal("Custom composites must provide exactly 3 wavelengths (e.g., 850,1650,660)")
        wanted.append(("CUSTOM", [float(x) for x in custom_wavelengths]))

    # Temp bands cache (1-based indices)
    temp_bands = {}
    created_names = []

    def ensure_band_written(idx1):
        if idx1 in temp_bands:
            return temp_bands[idx1]
        k = idx1 - 1
        band_EN = data[:, :, k].T.astype(np.float32)
        name = _temp_name(f"{output_name}_b{idx1:03d}")
        _write_float_raster(name, band_EN)
        temp_bands[idx1] = name
        created_names.append(name)
        return name

    # ------------------ peg region to a real raster FIRST ------------------
    # Prime enhanced RGB like PRISMA (write these bands now)
    rgb_target = COMPOSITES["RGB"]
    rgb_indices_1b = [_find_nearest_band_1based(w, wl) for w in rgb_target]
    for idx1 in rgb_indices_1b:
        ensure_band_written(idx1)

    # Use any of the prewritten RGB maps to peg region extents & resolution
    ref_map_for_region = next(iter({i: temp_bands[i] for i in rgb_indices_1b}.values()))
    Module("g.region", raster=ref_map_for_region, quiet=True)

    # Prepare a dict for reuse of enhanced RGB maps
    rgb_enhanced = {idx1: temp_bands[idx1] for idx1 in rgb_indices_1b}

    # -------------------------- 3D cube write --------------------------
    try:
        # depth only; XY extents/res already pegged to ref raster
        Module("g.region", b=0, t=bands_total, tbres=1, quiet=True)

        cube = garray.array3d(dtype=np.float32)
        for k in range(bands_total):
            cube[k, :, :] = data[:, :, k].T
        cube.write(mapname=f"{output_name}", null="nan", overwrite=True)  # NaNs -> NULLs
        gs.info(f"Created 3D raster cube with all bands: {output_name} ({bands_total} slices).")

        # r3 metadata (title + wavelengths/FWHM, like PRISMA)
        try:
            desc_lines = ["Hyperspectral Metadata:", f"Valid Bands: {bands_total}"]
            for i in range(bands_total):
                wl_i = float(wl[i])
                fwhm_i = float(fwhm[i]) if i < len(fwhm) else float("nan")
                desc_lines.append(f"Band {i+1}: {wl_i} nm, FWHM: {fwhm_i} nm")
            Module("r3.support",
                   map=output_name,
                   title="Tanager Hyperspectral Data",
                   description="\n".join(desc_lines),
                   vunit="nanometers",
                   quiet=True)
        except Exception as e_meta:
            gs.warning(f"Failed to write r3 metadata: {e_meta}")
    except Exception as e:
        gs.warning(f"3D cube creation failed: {e}")
    # ------------------------------------------------------------------

    # Composites
    for name, targets in wanted:
        bands_1b = [_find_nearest_band_1based(w, wl) for w in targets]
        maps = []
        for idx1 in bands_1b:
            maps.append(rgb_enhanced[idx1] if idx1 in rgb_enhanced else ensure_band_written(idx1))

        Module("g.region", raster=maps[0], quiet=True)
        if name.upper() == "RGB":
            Module("i.colors.enhance", red=maps[0], green=maps[1], blue=maps[2],
                   strength=str(strength_val), flags="p", quiet=True)
        else:
            Module("i.colors.enhance", red=maps[0], green=maps[1], blue=maps[2],
                   strength=str(strength_val), quiet=True)
        outname = f"{output_name}_{name.lower().replace('-', '_')}"
        Module("r.composite", red=maps[0], green=maps[1], blue=maps[2],
               output=outname, quiet=True, overwrite=True)
        gs.info(f"Generated composite raster: {outname}")

    if created_names:
        Module("g.remove", type="raster", name=",".join(created_names), flags="f", quiet=True)

    gs.del_temp_region()

# -------------------------- entry --------------------------
def run_import(options, flags):
    custom = None
    if options.get("composites_custom"):
        try:
            custom = [float(x.strip()) for x in options["composites_custom"].split(",")]
            if len(custom) != 3:
                raise ValueError
        except Exception:
            gs.fatal("Invalid format for composites_custom. Usage example: 850,1650,660")

    strength_opt = options.get("strength")
    if strength_opt is None or str(strength_opt).strip() == "":
        strength_val = 96
    else:
        try:
            strength_val = int(str(strength_opt).strip())
        except Exception:
            gs.fatal("Invalid strength. Provide an integer 0-100.")
        if not (0 <= strength_val <= 100):
            gs.fatal("Invalid strength. Provide an integer 0-100.")

    comps = [c.strip() for c in options["composites"].split(",")] if options.get("composites") else None
    import_null = bool(flags.get("n"))

    import_tanager(
        input_path=options["input"],
        output_name=options["output"],
        composites=comps,
        custom_wavelengths=custom,
        strength_val=strength_val,
        import_null=import_null,
    )