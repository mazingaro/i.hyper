#!/usr/bin/env python3
# Tanager → GRASS (3D + composites), PRISMA-style region handling
# - Entry: run_import() → import_tanager(...)
# - Writes float32 (SR or radiance) with NaNs outside footprint (NULL in GRASS)
# - Temp 2D bands for composites; cleans up afterwards
# - When orthorectify=True (default): forward bilinear splat + 8-neighbor fill (preserves nodata),
#   sets region from Planet_Ortho_Framing, writes full 3D cube + PRISMA-style r3 metadata.

import os
import uuid
import numpy as np
import grass.script as gs
import grass.script.array as garray
from grass.pygrass.modules import Module

from tanager_reader import (
    load_tanager_basic,
    read_planet_ortho_grid,
    orthorectify_band_to_planet_grid,
)

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
def import_tanager(input_path,
                   output_name,
                   composites=None,
                   custom_wavelengths=None,
                   strength_val=96,
                   import_null=False,
                   orthorectify=True):
    """
    If orthorectify=True (default):
      - read Planet_Ortho_Framing, set temp region,
      - orthorectify all bands to that grid (bilinear + local 8-neighbor fill),
      - write full 3D cube + r3 metadata,
      - produce composites from the same ortho bands.
    If orthorectify=False:
      - keep existing rectangular behaviour (no grid transform), also write 3D cube + metadata.
    """
    h5 = _resolve_h5(input_path)
    prod = load_tanager_basic(h5)

    data = prod.data              # (rows, cols, bands), float32 (NaNs where nodata_pixels==1)
    wl   = prod.wavelengths_nm
    fwhm = prod.fwhm_nm

    _require(data is not None and data.ndim == 3, "Tanager cube missing or invalid.")
    rows, cols, bands_total = data.shape

    # Build list of composites (default RGB)
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

    gs.use_temp_region()

    if orthorectify:
        _require(prod.lat is not None and prod.lon is not None, "Latitude/Longitude grids missing.")
        grid = read_planet_ortho_grid(h5)

        # Set computational region from Planet's ortho framing
        Module("g.region",
               w=grid.west, e=grid.east, s=grid.south, n=grid.north,
               ewres=grid.ewres, nsres=grid.nsres, flags="a", quiet=True)

        # On-demand band writer (orthorectify each band)
        def ensure_band_written(idx1):
            if idx1 in temp_bands:
                return temp_bands[idx1]
            k = idx1 - 1
            ortho2d = orthorectify_band_to_planet_grid(prod.lon, prod.lat, data[:, :, k], grid)
            name = _temp_name(f"{output_name}_b{idx1:03d}")
            _write_float_raster(name, ortho2d)
            temp_bands[idx1] = name
            created_names.append(name)
            return name

        # Build ORTHO 3D cube (bands in Z)
        try:
            # mirror 2D res to 3D
            reg2d = gs.region()
            nsres2d = float(reg2d["nsres"])
            ewres2d = float(reg2d["ewres"])

            # Set 3D region (Z as band index 1..bands_total)
            Module("g.region", nsres3=nsres2d, ewres3=ewres2d, b=0, t=bands_total, tbres=1, quiet=True)

            cube = garray.array3d(dtype=np.float32)
            for k in range(bands_total):
                # orthorectify & write into the cube
                ortho2d = orthorectify_band_to_planet_grid(prod.lon, prod.lat, data[:, :, k], grid)
                cube[k, :, :] = ortho2d

            cube.write(mapname=f"{output_name}", null="nan", overwrite=True)
            gs.info(f"Created ORTHO 3D raster cube with all bands: {output_name} ({bands_total} slices).")

            # r3 metadata (like PRISMA)
            try:
                desc_lines = ["Hyperspectral Metadata:", f"Valid Bands: {bands_total}"]
                for i in range(bands_total):
                    wl_i = float(wl[i])
                    fwhm_i = float(fwhm[i]) if i < len(fwhm) else float("nan")
                    desc_lines.append(f"Band {i+1}: {wl_i} nm, FWHM: {fwhm_i} nm")
                Module("r3.support",
                       map=output_name,
                       title="Tanager Hyperspectral Data (Orthorectified)",
                       description="\n".join(desc_lines),
                       vunit="nanometers",
                       quiet=True)
            except Exception as e_meta:
                gs.warning(f"Failed to write r3 metadata: {e_meta}")
        except Exception as e:
            gs.warning(f"3D cube creation (ortho) failed: {e}")

        rgb_enhanced = {}  # region already fixed; we don't need to prewrite RGB here

    else:
        # ---- existing rectangular (image-space) path ----
        # Determine transposed shape (E,N) from any band and set rows/cols
        first_band = data[:, :, 0].T
        rows_E, cols_N = first_band.shape
        Module("g.region", rows=rows_E, cols=cols_N, quiet=True)

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

        # Prewrite RGB to peg region & cache
        rgb_target = COMPOSITES["RGB"]
        rgb_indices_1b = [_find_nearest_band_1based(w, wl) for w in rgb_target]
        for idx1 in rgb_indices_1b:
            ensure_band_written(idx1)
        ref_map_for_region = next(iter({i: temp_bands[i] for i in rgb_indices_1b}.values()))
        Module("g.region", raster=ref_map_for_region, quiet=True)
        rgb_enhanced = {idx1: temp_bands[idx1] for idx1 in rgb_indices_1b}

        # Build 3D cube from original (non-ortho) data for parity with PRISMA
        try:
            Module("g.region", b=0, t=bands_total, tbres=1, quiet=True)
            cube = garray.array3d(dtype=np.float32)
            for k in range(bands_total):
                cube[k, :, :] = data[:, :, k].T
            cube.write(mapname=f"{output_name}", null="nan", overwrite=True)
            gs.info(f"Created 3D raster cube with all bands: {output_name} ({bands_total} slices).")

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

    # -------------------------- composites (same as PRISMA flow) --------------------------
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
        orthorectify=True
    )
