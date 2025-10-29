#!/usr/bin/env python3
# PRISMA L2D → GRASS (2D reflectance; composite-only, EnMAP-style composite block)
# - Same entrypoints: run_import() -> import_prisma(...)
# - Writes float32 REFLECTANCE (0..1) to TEMP band maps, builds composites like EnMAP code does, then deletes temps
# - Transpose to (E,N) before writing; region rows=E, cols=N

import os
import uuid
import numpy as np
import grass.script as gs
import grass.script.array as garray
from grass.pygrass.modules import Module

from prisma_importer import load_prisma_l2d, concatenate_hyperspectral


# ---- Inlined importer code (was a separate module) ----
#!/usr/bin/env python3
"""
PRISMA L2D importer

- Reads VNIR/SWIR/PAN cubes + error matrices from PRISMA L2D HDF-EOS5.
- Converts DN->reflectance on demand using product's L2 scale attributes:
    refl = Min + (DN * (Max - Min)) / 65535
- Extracts wavelengths/FWHM from global attributes and filters by *_Flags==1.
- Normalizes VNIR/SWIR arrays to (rows, cols, bands) with bands-last.
- Exposes per-pixel lat/lon grids and scalar corner easting/northing attributes in meters.

Spec assumptions (strict):
- VNIR/SWIR data cubes have dimensions (nEastingPixel, nBands, nNorthingPixel).
- Latitude/Longitude are (nEastingPixel, nNorthingPixel).
- EPSG code is stored in global attribute 'Epsg_Code'.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
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


# ---- HDF5 paths (from PRISMA L2D spec) ----
HCO_VNIR_DATA = "/HDFEOS/SWATHS/PRS_L2D_HCO/Data Fields/VNIR_Cube"
HCO_SWIR_DATA = "/HDFEOS/SWATHS/PRS_L2D_HCO/Data Fields/SWIR_Cube"
PCO_PAN_DATA  = "/HDFEOS/SWATHS/PRS_L2D_PCO/Data Fields/PAN_Cube"

HCO_VNIR_ERR  = "/HDFEOS/SWATHS/PRS_L2D_HCO/Data Fields/VNIR_PIXEL_L2_ERR_MATRIX"
HCO_SWIR_ERR  = "/HDFEOS/SWATHS/PRS_L2D_HCO/Data Fields/SWIR_PIXEL_L2_ERR_MATRIX"
PCO_PAN_ERR   = "/HDFEOS/SWATHS/PRS_L2D_PCO/Data Fields/PIXEL_L2_ERR_MATRIX"

HCO_LAT = "/HDFEOS/SWATHS/PRS_L2D_HCO/Geolocation Fields/Latitude"
HCO_LON = "/HDFEOS/SWATHS/PRS_L2D_HCO/Geolocation Fields/Longitude"
PCO_LAT = "/HDFEOS/SWATHS/PRS_L2D_PCO/Geolocation Fields/Latitude"
PCO_LON = "/HDFEOS/SWATHS/PRS_L2D_PCO/Geolocation Fields/Longitude"

# ---- Global attributes (per spec) ----
ATTR_CW_VNIR       = "List_Cw_Vnir"
ATTR_CW_VNIR_FLAGS = "List_Cw_Vnir_Flags"
ATTR_FWHM_VNIR     = "List_Fwhm_Vnir"

ATTR_CW_SWIR       = "List_Cw_Swir"
ATTR_CW_SWIR_FLAGS = "List_Cw_Swir_Flags"
ATTR_FWHM_SWIR     = "List_Fwhm_Swir"

ATTR_SCALE_VMAX = "L2ScaleVnirMax"
ATTR_SCALE_VMIN = "L2ScaleVnirMin"
ATTR_SCALE_SMAX = "L2ScaleSwirMax"
ATTR_SCALE_SMIN = "L2ScaleSwirMin"
ATTR_SCALE_PMAX = "L2ScalePanMax"
ATTR_SCALE_PMIN = "L2ScalePanMin"

ATTR_CENTER_E   = "Product_center_easting"
ATTR_CENTER_N   = "Product_center_northing"

ATTR_LL_E = "Product_LLcorner_easting"
ATTR_LL_N = "Product_LLcorner_northing"
ATTR_LR_E = "Product_LRcorner_easting"
ATTR_LR_N = "Product_LRcorner_northing"
ATTR_UL_E = "Product_ULcorner_easting"
ATTR_UL_N = "Product_ULcorner_northing"
ATTR_UR_E = "Product_URcorner_easting"
ATTR_UR_N = "Product_URcorner_northing"

ATTR_EPSG = "Epsg_Code"  # strict per spec

# ---- Data containers ----
@dataclass
class BandInfo:
    wavelengths_nm: np.ndarray     # (bands_kept,)
    fwhm_nm: np.ndarray            # (bands_kept,)
    present_flags: np.ndarray      # (bands_kept,) all ones after filtering
    # Added: indices of kept bands in the original band axis (0-based)
    kept_indices: np.ndarray       # (bands_kept,)

@dataclass
class PrismaCube:
    name: str                          # "VNIR" | "SWIR" | "PAN"
    dn: Optional[np.ndarray]           # VNIR/SWIR: (rows, cols, bands); PAN: (rows, cols)
    err: Optional[np.ndarray]          # same spatial shape; band-dim if provided
    scale_min: Optional[float]
    scale_max: Optional[float]
    bands: Optional[BandInfo] = None   # None for PAN

    def to_reflectance(self) -> Optional[np.ndarray]:
        if self.dn is None or self.scale_min is None or self.scale_max is None:
            return None
        return self.scale_min + (self.dn.astype(np.float32) *
                                 (self.scale_max - self.scale_min) / 65535.0)

    def valid_mask(self) -> Optional[np.ndarray]:
        return (self.err == 0) if self.err is not None else None

@dataclass
class Geolocation:
    lat: np.ndarray                 # (rows, cols)
    lon: np.ndarray                 # (rows, cols)
    x_m: Optional[np.ndarray]       # not computed here
    y_m: Optional[np.ndarray]
    utm_epsg: Optional[int]
    center_e: Optional[float] = None
    center_n: Optional[float] = None
    ll_e: Optional[float] = None
    ll_n: Optional[float] = None
    lr_e: Optional[float] = None
    lr_n: Optional[float] = None
    ul_e: Optional[float] = None
    ul_n: Optional[float] = None
    ur_e: Optional[float] = None
    ur_n: Optional[float] = None

@dataclass
class PrismaL2DProduct:
    path: str
    vnir: Optional[PrismaCube]
    swir: Optional[PrismaCube]
    pan: Optional[PrismaCube]
    hco_geo: Optional[Geolocation]
    pco_geo: Optional[Geolocation]
    attrs: Dict[str, Any]

# ---- Internal helpers ----
def _read_attr_as_array(attrs: h5py.AttributeManager, key: str) -> Optional[np.ndarray]:
    if key not in attrs:
        return None
    v = attrs[key]
    if isinstance(v, (bytes, bytearray, str)):
        try:
            s = v.decode() if isinstance(v, (bytes, bytearray)) else v
            parts = [p for p in s.replace(",", " ").split() if p.strip()]
            return np.array([float(p) for p in parts], dtype=np.float32) if parts else None
        except Exception:
            return None
    arr = np.array(v)
    if arr.dtype.kind in ("i", "u", "f"):
        return arr.astype(np.float32)
    if arr.dtype.kind == "S":
        try:
            return np.array([x.decode() for x in arr], dtype=np.float32)
        except Exception:
            return None
    return None

def _read_attr_scalar(attrs: h5py.AttributeManager, key: str) -> Optional[float]:
    if key not in attrs:
        return None
    v = attrs[key]
    if isinstance(v, (np.generic, np.ndarray)):
        v = np.array(v).squeeze().tolist()
    if isinstance(v, (bytes, bytearray)):
        try:
            v = float(v.decode())
        except Exception:
            return None
    try:
        return float(v)
    except Exception:
        return None

def _select_present_bands(cw: np.ndarray, fwhm: np.ndarray, flags: np.ndarray):
    flags = flags.astype(int).ravel()
    idx = np.where(flags == 1)[0]  # 0-based indices into the original band axis
    return cw[idx], fwhm[idx], idx

def _maybe_read(f: h5py.File, path: str) -> Optional[np.ndarray]:
    return f[path][()] if path in f else None

def _load_bandinfo_from_attrs(attrs: h5py.AttributeManager,
                              cw_key: str, fwhm_key: str, flags_key: str) -> Optional[BandInfo]:
    cw = _read_attr_as_array(attrs, cw_key)
    fwhm = _read_attr_as_array(attrs, fwhm_key)
    flags = _read_attr_as_array(attrs, flags_key)
    if cw is None or fwhm is None or flags is None:
        return None
    cw_sel, fwhm_sel, kept_idx = _select_present_bands(cw, fwhm, flags)
    return BandInfo(
        wavelengths_nm=cw_sel,
        fwhm_nm=fwhm_sel,
        present_flags=np.ones_like(cw_sel, dtype=np.uint8),
        kept_indices=kept_idx.astype(np.int64),
    )

def _read_corners(attrs: h5py.AttributeManager) -> Dict[str, Optional[float]]:
    keys = [ATTR_CENTER_E, ATTR_CENTER_N,
            ATTR_LL_E, ATTR_LL_N, ATTR_LR_E, ATTR_LR_N,
            ATTR_UL_E, ATTR_UL_N, ATTR_UR_E, ATTR_UR_N]
    return {k: _read_attr_scalar(attrs, k) for k in keys}

# Fixed, spec-driven: (E, B, N) -> (N, E, B)
def _l2d_bil_to_rows_cols_bands(arr: np.ndarray) -> np.ndarray:
    """
    PRISMA L2D VNIR/SWIR cubes are (nEastingPixel, nBands, nNorthingPixel).
    Return array as (rows=N, cols=E, bands=B) i.e. np.transpose(arr, (2, 0, 1)).
    """
    if arr.ndim != 3:
        raise ValueError(f"L2D cube must be 3D (E,B,N); got {arr.shape}")
    return np.transpose(arr, (2, 0, 1))

# ---- Public API ----
def load_prisma_l2d(product_path: str,
                    load_pan: bool = False,
                    compute_utm: bool = False  # kept for signature compatibility; not used
                    ) -> PrismaL2DProduct:
    with h5py.File(product_path, "r") as f:
        # Global attrs (kept for reference)
        attrs: Dict[str, Any] = {}
        for k, v in f.attrs.items():
            try:
                attrs[k] = v.decode(errors="ignore") if isinstance(v, (bytes, bytearray)) else (
                    np.array(v).tolist() if isinstance(v, np.ndarray) else v
                )
            except Exception:
                attrs[k] = v

        # Geolocation grids (hyperspectral swath)
        lat_hco = _maybe_read(f, HCO_LAT)
        lon_hco = _maybe_read(f, HCO_LON)

        # EPSG from spec key
        epsg_meta = int(_read_attr_scalar(f.attrs, ATTR_EPSG)) if _read_attr_scalar(f.attrs, ATTR_EPSG) is not None else None

        # VNIR band metadata
        vnir_bands = _load_bandinfo_from_attrs(f.attrs, ATTR_CW_VNIR, ATTR_FWHM_VNIR, ATTR_CW_VNIR_FLAGS)
        # SWIR band metadata
        swir_bands = _load_bandinfo_from_attrs(f.attrs, ATTR_CW_SWIR, ATTR_FWHM_SWIR, ATTR_CW_SWIR_FLAGS)

        # VNIR data
        vnir = None
        vnir_dn_raw = _maybe_read(f, HCO_VNIR_DATA)
        vnir_err_raw = _maybe_read(f, HCO_VNIR_ERR)
        if vnir_dn_raw is not None:
            vnir_dn = _l2d_bil_to_rows_cols_bands(vnir_dn_raw)
            vnir_err = _l2d_bil_to_rows_cols_bands(vnir_err_raw) if (vnir_err_raw is not None and vnir_err_raw.ndim == 3) else vnir_err_raw
            vnir = PrismaCube(
                name="VNIR",
                dn=vnir_dn,
                err=vnir_err,
                scale_min=_read_attr_scalar(f.attrs, ATTR_SCALE_VMIN),
                scale_max=_read_attr_scalar(f.attrs, ATTR_SCALE_VMAX),
                bands=vnir_bands,
            )

        # SWIR data
        swir = None
        swir_dn_raw = _maybe_read(f, HCO_SWIR_DATA)
        swir_err_raw = _maybe_read(f, HCO_SWIR_ERR)
        if swir_dn_raw is not None:
            swir_dn = _l2d_bil_to_rows_cols_bands(swir_dn_raw)
            swir_err = _l2d_bil_to_rows_cols_bands(swir_err_raw) if (swir_err_raw is not None and swir_err_raw.ndim == 3) else swir_err_raw
            swir = PrismaCube(
                name="SWIR",
                dn=swir_dn,
                err=swir_err,
                scale_min=_read_attr_scalar(f.attrs, ATTR_SCALE_SMIN),
                scale_max=_read_attr_scalar(f.attrs, ATTR_SCALE_SMAX),
                bands=swir_bands,
            )

        # HCO geolocation (VNIR/SWIR)
        hco_geo = None
        if lat_hco is not None and lon_hco is not None:
            corners = _read_corners(f.attrs)
            hco_geo = Geolocation(
                lat=lat_hco, lon=lon_hco, x_m=None, y_m=None, utm_epsg=epsg_meta,
                center_e=corners.get(ATTR_CENTER_E), center_n=corners.get(ATTR_CENTER_N),
                ll_e=corners.get(ATTR_LL_E), ll_n=corners.get(ATTR_LL_N),
                lr_e=corners.get(ATTR_LR_E), lr_n=corners.get(ATTR_LR_N),
                ul_e=corners.get(ATTR_UL_E), ul_n=corners.get(ATTR_UL_N),
                ur_e=corners.get(ATTR_UR_E), ur_n=corners.get(ATTR_UR_N),
            )

        # PAN (optional; PAN array is already 2D)
        pan = None
        pco_geo = None
        if load_pan:
            pan_dn = _maybe_read(f, PCO_PAN_DATA)
            pan_err = _maybe_read(f, PCO_PAN_ERR)
            if pan_dn is not None:
                pan = PrismaCube(
                    name="PAN", dn=pan_dn, err=pan_err,
                    scale_min=_read_attr_scalar(f.attrs, ATTR_SCALE_PMIN),
                    scale_max=_read_attr_scalar(f.attrs, ATTR_SCALE_PMAX),
                    bands=None
                )
            lat_pco = _maybe_read(f, PCO_LAT)
            lon_pco = _maybe_read(f, PCO_LON)
            if lat_pco is not None and lon_pco is not None:
                corners = _read_corners(f.attrs)
                pco_geo = Geolocation(
                    lat=lat_pco, lon=lon_pco, x_m=None, y_m=None, utm_epsg=epsg_meta,
                    center_e=corners.get(ATTR_CENTER_E), center_n=corners.get(ATTR_CENTER_N),
                    ll_e=corners.get(ATTR_LL_E), ll_n=corners.get(ATTR_LL_N),
                    lr_e=corners.get(ATTR_LR_E), lr_n=corners.get(ATTR_LR_N),
                    ul_e=corners.get(ATTR_UL_E), ul_n=corners.get(ATTR_UL_N),
                    ur_e=corners.get(ATTR_UR_E), ur_n=corners.get(ATTR_UR_N),
                )

    return PrismaL2DProduct(
        path=product_path,
        vnir=vnir,
        swir=swir,
        pan=pan,
        hco_geo=hco_geo,
        pco_geo=pco_geo,
        attrs=attrs,
    )

def concatenate_hyperspectral(product: PrismaL2DProduct
                              ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Concatenate VNIR and SWIR reflectance along band axis (bands-last), **after filtering**
    to only the bands marked present (flags==1). This keeps the reflectance cube and the
    metadata arrays (wavelengths, FWHM) perfectly aligned.
    Returns:
        refl (rows, cols, bands_total_filtered),
        wavelengths_nm (bands_total_filtered,),
        fwhm_nm (bands_total_filtered,)
    """
    if product.vnir is None or product.swir is None:
        raise ValueError("Both VNIR and SWIR must be present to concatenate.")
    vnir_ref = product.vnir.to_reflectance()
    swir_ref = product.swir.to_reflectance()
    if vnir_ref is None or swir_ref is None:
        raise ValueError("Missing scale factors to compute reflectance.")
    if vnir_ref.ndim != 3 or swir_ref.ndim != 3:
        raise ValueError(f"Expected 3D arrays; got VNIR {vnir_ref.shape}, SWIR {swir_ref.shape}")
    if vnir_ref.shape[:2] != swir_ref.shape[:2]:
        raise ValueError(f"Spatial shapes differ after normalization: VNIR {vnir_ref.shape[:2]} vs SWIR {swir_ref.shape[:2]}")

    # ---- Filter by kept indices (flags==1) so band counts match metadata ----
    if product.vnir.bands is None or product.swir.bands is None:
        raise ValueError("Missing wavelength/FWHM metadata.")
    v_idx = product.vnir.bands.kept_indices
    s_idx = product.swir.bands.kept_indices
    vnir_ref_f = vnir_ref[:, :, v_idx]
    swir_ref_f = swir_ref[:, :, s_idx]

    v_wl   = product.vnir.bands.wavelengths_nm
    s_wl   = product.swir.bands.wavelengths_nm
    v_fwhm = product.vnir.bands.fwhm_nm
    s_fwhm = product.swir.bands.fwhm_nm

    refl = np.concatenate([vnir_ref_f, swir_ref_f], axis=2).astype(np.float32)
    wavelengths = np.concatenate([v_wl, s_wl], axis=0)
    fwhm = np.concatenate([v_fwhm, s_fwhm], axis=0)
    return refl, wavelengths, fwhm
# ---- End inlined importer code ----


COMPOSITES = {
    "RGB":              [660.0, 572.0, 478.0],
    "CIR":              [848.0, 660.0, 572.0],
    "SWIR-agriculture": [848.0, 1653.0, 660.0],
    "SWIR-geology":     [2200.0, 848.0, 572.0],
}

# -------------------------- helpers --------------------------

def _resolve_he5(path_like):
    if os.path.isdir(path_like):
        for n in os.listdir(path_like):
            if n.lower().endswith(".he5"):
                return os.path.join(path_like, n)
        gs.fatal("No .he5 file found in the provided folder.")
    return path_like



# -------------------------- region --------------------------
def _compute_edges_from_centers(ul_e, ul_n, ur_e, ur_n, ll_e, ll_n, rows, cols):
    """
    Edges from pixel-center corners.
    Called with rows=E (UL→LL samples) and cols=N (UL→UR samples) AFTER transposing.
    """
    if any(v is None for v in (ul_e, ul_n, ur_e, ur_n, ll_e, ll_n)):
        gs.fatal("PRISMA corners missing (need UL, UR, LL).")
    if rows < 2 or cols < 2:
        gs.fatal("Invalid raster shape (<2).")

    ew_c2c = (ur_e - ul_e) / float(cols - 1)   # columns axis = easting
    ns_c2c = (ul_n - ll_n) / float(rows - 1)   # rows axis = northing

    west  = ul_e - 0.5 * ew_c2c
    east  = ur_e + 0.5 * ew_c2c
    north = ul_n + 0.5 * ns_c2c
    south = ll_n - 0.5 * ns_c2c
    return west, east, south, north

def _force_region_exact_for_transposed(geo, rows_E, cols_N):
    west, east, south, north = _compute_edges_from_centers(
        geo.ul_e, geo.ul_n, geo.ur_e, geo.ur_n, geo.ll_e, geo.ll_n,
        rows=rows_E, cols=cols_N
    )
    gs.run_command("g.region", w=west, e=east, s=south, n=north, quiet=True)
    gs.run_command("g.region", rows=rows_E, cols=cols_N, quiet=True)
    reg = gs.region()
    if int(reg["rows"]) != rows_E or int(reg["cols"]) != cols_N:
        gs.fatal(f"Region is {reg['rows']}x{reg['cols']} but transposed data is {rows_E}x{cols_N}")

# -------------------------- writers --------------------------
def _write_float_raster(name, data_2d_float32):
    arr = garray.array(dtype=np.float32)
    arr[:, :] = data_2d_float32
    arr.write(name, overwrite=True)

# -------------------------- public core --------------------------
def import_prisma(input_path, output_name, composites=None, custom_wavelengths=None, strength_val=96, import_null=False):
    """
    Writes composite rasters (only) following the EnMAP composite flow:
      - pick nearest bands by wavelength,
      - enhance RGB with -p flag, others without,
      - build composite from three band maps,
      - only final composites remain (temp bands removed).
    Reflectance is float32; per-band temp rasters are created on demand and reused across composites.
    """
    he5 = _resolve_he5(input_path)
    prod = load_prisma_l2d(he5, load_pan=False, compute_utm=False)

    _require(prod.hco_geo is not None, "HCO geolocation missing.")
    _require(prod.vnir and prod.vnir.dn is not None, "VNIR cube missing.")
    _require(prod.swir and prod.swir.dn is not None, "SWIR cube missing.")

    # Reflectance cube (N,E,B) and wavelengths
    # NOTE: capturing fwhm as well for r3 metadata compatibility with EnMAP
    refl, wavelengths, fwhm = concatenate_hyperspectral(prod)  # float32 0..1
    _require(refl.ndim == 3, f"Unexpected reflectance shape: {refl.shape}")

    # Determine transposed shape (E,N) from any band
    first_band = refl[:, :, 0].T                # (E,N)
    rows_E, cols_N = first_band.shape

    # Fix region exactly to transposed shape and metadata extents
    gs.use_temp_region()
    _force_region_exact_for_transposed(prod.hco_geo, rows_E, cols_N)

    # Build list of composites to make (default RGB)
    wanted = []
    if composites:
        for comp in composites:
            compu = comp.strip().upper()
            if compu in COMPOSITES:
                wanted.append((compu, COMPOSITES[compu]))
    else:
        wanted.append(("RGB", COMPOSITES["RGB"]))

    if custom_wavelengths:
        if len(custom_wavelengths) != 3:
            gs.fatal("Custom composites must provide exactly 3 wavelengths (e.g., 850,1650,660)")
        wanted.append(("CUSTOM", [float(x) for x in custom_wavelengths]))

    # --- EnMAP-style band handling:
    # We'll create temp rasters only for bands we need, and reuse them via a dict keyed by 1-based band index.
    temp_bands = {}  # {band_idx_1based: raster_name}
    created_names = []  # for final cleanup

    def ensure_band_written(idx1):
        """Write reflectance band idx1 (1-based) as a temp raster (E,N) if not already created."""
        if idx1 in temp_bands:
            return temp_bands[idx1]
        # Extract band; refl is (N,E,B) with 0-based k
        k = idx1 - 1
        band_EN = refl[:, :, k].T.astype(np.float32)
        name = _temp_name(f"{output_name}_b{idx1:03d}")
        _write_float_raster(name, band_EN)
        temp_bands[idx1] = name
        created_names.append(name)
        return name

    # Prime the "rgb_enhanced" mapping exactly like EnMAP:
    rgb_target = COMPOSITES["RGB"]
    rgb_indices_1b = [_find_nearest_band_1based(w, wavelengths) for w in rgb_target]
    # Create these bands now and cache
    for idx1 in rgb_indices_1b:
        ensure_band_written(idx1)
    rgb_enhanced = {idx1: temp_bands[idx1] for idx1 in rgb_indices_1b}

    # -------------------------- build full hyperspectral 3D cube (all bands) --------------------------
    try:
        bands_total = int(refl.shape[2])

        # 1) Peg 2D region to an existing temp band (guarantees XY extents & 2D res match slices)
        ref_map_for_region = next(iter(rgb_enhanced.values()))
        Module("g.region", raster=ref_map_for_region, quiet=True)

        # 2) Read the (now pegged) 2D region to mirror its XY resolutions into the 3D region
        reg2d = gs.region()
        nsres2d = float(reg2d["nsres"])
        ewres2d = float(reg2d["ewres"])

        # -------- spectral (Z) axis in nanometers like EnMAP --------
        if wavelengths is not None and len(wavelengths) > 0:
            wl = np.asarray(wavelengths, dtype=float)
            if wl.size > 1:
                # Use exact spacing from endpoints (sum of diffs) to avoid accumulating FP error
                tbres_nm = float((wl[-1] - wl[0]) / (bands_total - 1))
            else:
                tbres_nm = 1.0
            bottom_nm = float(wl[0])
            # IMPORTANT: GRASS depth behaves like depth = int((t - b) / tbres)
            # so we set t = b + tbres * bands_total to get depth == bands_total
            top_nm = bottom_nm + tbres_nm * bands_total  # <-- fix off-by-one
        else:
            bottom_nm, top_nm, tbres_nm = 0.0, float(bands_total), 1.0

        # 3) Set ONLY 3D params: mirror XY resolutions and define Z (bottom/top/tbres)
        #    NOTE: rows3/cols3 DO NOT EXIST; they are derived from extents + nsres3/ewres3.
        gs.run_command(
            "g.region",
            nsres3=nsres2d,
            ewres3=ewres2d,
            b=0,
            t=bands_total,
            tbres=1,
            quiet=True,
        )

        # 4) Create and fill the 3D array: (band, row(E), col(N))
        cube = garray.array3d(dtype=np.float32)
        for k in range(bands_total):
            cube[k, :, :] = refl[:, :, k].T.astype(np.float32)

        # write 3D raster under the final output name (compat with EnMAP)
        cube.write(mapname=f"{output_name}", overwrite=True)
        gs.info(f"Created 3D raster cube with all bands: {output_name} ({bands_total} slices).")

        # -------- r3 metadata to match EnMAP's r3.support pattern --------
        try:
            count_meta = int(min(bands_total, len(wavelengths)))
            desc_lines = ["Hyperspectral Metadata:", f"Valid Bands: {count_meta}"]
            # Use 1-based band indices like EnMAP metadata
            for i in range(count_meta):
                wl_i = float(wavelengths[i])
                fwhm_i = float(fwhm[i]) if i < len(fwhm) else float("nan")
                desc_lines.append(f"Band {i+1}: {wl_i} nm, FWHM: {fwhm_i} nm")
            Module("r3.support",
                   map=output_name,
                   title="PRISMA Hyperspectral Data",
                   description="\n".join(desc_lines),
                   vunit="nanometers",
                   quiet=True)
        except Exception as e_meta:
            gs.warning(f"Failed to write r3 metadata: {e_meta}")
        # -----------------------------------------------------------------
    except Exception as e:
        gs.warning(f"3D cube creation failed: {e}")
    # -------------------------------------------------------------------------------------------------------

    # For each requested composite, select bands and build r.composite
    for name, targets in wanted:
        bands_1b = [_find_nearest_band_1based(w, wavelengths) for w in targets]
        # EnMAP logic: reuse enhanced RGB maps when available, otherwise fallback to band rasters
        rgb_maps = []
        for idx1 in bands_1b:
            if idx1 in rgb_enhanced:
                rgb_maps.append(rgb_enhanced[idx1])
            else:
                rgb_maps.append(ensure_band_written(idx1))

        # EnMAP: set region to first map and enhance (RGB with -p, others normal)
        Module("g.region", raster=rgb_maps[0], quiet=True)
        if name.upper() == "RGB":
            Module("i.colors.enhance", red=rgb_maps[0], green=rgb_maps[1], blue=rgb_maps[2],
                   strength=str(strength_val), flags="p", quiet=True)
            outname = f"{output_name}_{name.lower().replace('-', '_')}"
        else:
            Module("i.colors.enhance", red=rgb_maps[0], green=rgb_maps[1], blue=rgb_maps[2],
                   strength=str(strength_val), quiet=True)
            outname = f"{output_name}_{name.lower().replace('-', '_')}"

        Module("r.composite", red=rgb_maps[0], green=rgb_maps[1], blue=rgb_maps[2],
               output=outname, quiet=True, overwrite=True)
        gs.info(f"Generated composite raster: {outname}")

    # Clean up temp bands after all composites are made
    if created_names:
        Module("g.remove", type="raster", name=",".join(created_names), flags="f", quiet=True)

    gs.del_temp_region()

# -------------------------- entry (same) --------------------------
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

    import_prisma(
        input_path=options["input"],
        output_name=options["output"],
        composites=comps,
        custom_wavelengths=custom,
        strength_val=strength_val,
        import_null=import_null,
    )