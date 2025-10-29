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

from tanager_reader import load_tanager_basic

COMPOSITES = {
    "RGB":              [660.0, 572.0, 478.0],
    "CIR":              [848.0, 660.0, 572.0],
    "SWIR-agriculture": [848.0, 1653.0, 660.0],
    "SWIR-geology":     [2200.0, 848.0, 572.0],
}
# -------------------------- helpers --------------------------
def _require(cond, msg):
    if not cond:
        gs.fatal(msg)

def _resolve_h5(path_like):
    if os.path.isdir(path_like):
        for n in os.listdir(path_like):
            if n.lower().endswith(".h5"):
                return os.path.join(path_like, n)
        gs.fatal("No .h5 file found in the provided folder.")
    return path_like

def _find_nearest_band_1based(target_nm, wavelengths_nm):
    wl = np.asarray(wavelengths_nm, dtype=np.float32)
    return int(np.argmin(np.abs(wl - float(target_nm)))) + 1  # 1-based

def _temp_name(prefix):
    return f"{prefix}_{uuid.uuid4().hex[:8]}"

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
