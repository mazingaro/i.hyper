#!/usr/bin/env python3
"""
Tanager BASIC reader (radiance or surface_reflectance)

- Reads (Band, Y, X) cube and transposes to (rows=Y, cols=X, bands).
- Extracts wavelengths/FWHM from dataset attributes.
- Applies outside-footprint mask using /HDFEOS/SWATHS/HYP/Data Fields/nodata_pixels (1 -> NaN).
- Exposes per-pixel lat/lon grids.
- Ensures ascending wavelength order (and reorders cube accordingly).
- Provides PRISMA-style orthorectify helpers targeting Planet_Ortho_Framing.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple
import numpy as np
import h5py
import json

# Optional deps (GRASS envs typically have pyproj; SciPy may or may not be present)
try:
    from pyproj import CRS, Transformer
    _HAS_PYPROJ = True
except Exception:
    _HAS_PYPROJ = False

try:
    from scipy.ndimage import distance_transform_edt  # for local NN fill
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

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

        arr_raw = dset[()]  # (Band, Y, X)
        if arr_raw.ndim != 3:
            raise ValueError(f"Unexpected Tanager dims {arr_raw.shape} (expected 3D)")

        # to (rows, cols, bands)
        data = np.transpose(arr_raw, (1, 2, 0)).astype(np.float32, copy=False)

        # spectral meta
        wl   = np.array(dset.attrs[DS_WL_ATTR], dtype=np.float32)
        fwhm = np.array(dset.attrs.get(DS_FWHM_ATTR, np.full_like(wl, np.nan, dtype=np.float32)), dtype=np.float32)

        # ensure ascending wavelength order
        order = np.argsort(wl.astype(np.float32))
        wl = wl[order]
        fwhm = fwhm[order]
        data = data[:, :, order]

        # geolocation + nodata mask
        lat = _maybe(f, DS_LAT)
        lon = _maybe(f, DS_LON)

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

# -------------------- PRISMA-style orthorectify helpers --------------------

@dataclass(frozen=True)
class OrthoGrid:
    """Planet ortho grid definition (UTM @ 30 m)."""
    epsg: int
    west: float     # GT[0]
    north: float    # GT[3]
    ewres: float    # +pixel width (m)
    nsres: float    # +pixel height (m)
    rows: int
    cols: int
    @property
    def east(self) -> float:
        return self.west + self.cols * self.ewres
    @property
    def south(self) -> float:
        return self.north - self.rows * self.nsres

def read_planet_ortho_grid(product_path: str) -> OrthoGrid:
    """
    Read Planet's framing JSON from HDF5 attribute:
      /HDFEOS/SWATHS/HYP/Geolocation Fields : Planet_Ortho_Framing
      {"epsg_code": 32658, "rows": 840, "cols": 877,
       "geotransform": [west, ewres, 0, north, 0, -nsres]}
    """
    with h5py.File(product_path, "r") as f:
        meta = f[f"{GF}"].attrs["Planet_Ortho_Framing"]
        if isinstance(meta, (bytes, bytearray)):
            meta = meta.decode(errors="ignore")
        meta = json.loads(meta)
    epsg = int(meta["epsg_code"])
    rows = int(meta["rows"])
    cols = int(meta["cols"])
    gt = meta["geotransform"]
    west, ewres, north, nsres = float(gt[0]), float(gt[1]), float(gt[3]), float(-gt[5])
    return OrthoGrid(epsg, west, north, ewres, nsres, rows, cols)

def _transform_lonlat(lon: np.ndarray, lat: np.ndarray, epsg: int) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorized WGS84 lon/lat -> target EPSG (meters)."""
    if not _HAS_PYPROJ:
        raise RuntimeError("pyproj is required for in-memory orthorectification.")
    t = Transformer.from_crs(CRS.from_epsg(4326), CRS.from_epsg(epsg), always_xy=True)
    x, y = t.transform(lon, lat)
    return np.asarray(x), np.asarray(y)

def _splat_bilinear(values, x, y, grid: OrthoGrid, nodata=np.nan):
    """
    Forward bilinear splatting with masks to preserve nodata.
    Returns (ortho_f32, wts, visit, vnod).
    """
    rows, cols = grid.rows, grid.cols
    out  = np.zeros((rows, cols), dtype=np.float64)   # sum of valid*weight
    wts  = np.zeros((rows, cols), dtype=np.float64)   # sum of weights for valid
    visit = np.zeros((rows, cols), dtype=np.float64)  # any sample (valid or nodata)
    vnod  = np.zeros((rows, cols), dtype=np.float64)  # nodata-only indicator

    fx = (x - grid.west) / grid.ewres
    fy = (grid.north - y) / grid.nsres
    c0 = np.floor(fx).astype(np.int64)
    r0 = np.floor(fy).astype(np.int64)
    dc = fx - c0
    dr = fy - r0

    valid = np.isfinite(values) & np.isfinite(fx) & np.isfinite(fy)
    nod   = ~np.isfinite(values) & np.isfinite(fx) & np.isfinite(fy)

    # weights for the 4 neighbors
    w00 = (1 - dr) * (1 - dc)
    w10 = dr * (1 - dc)
    w01 = (1 - dr) * dc
    w11 = dr * dc

    def add(acc, rr, cc, ww, mask=None, scale=None):
        m = (rr >= 0) & (rr < rows) & (cc >= 0) & (cc < cols) & (ww > 0)
        if mask is not None:
            m &= mask
        if np.any(m):
            if scale is None:
                np.add.at(acc, (rr[m], cc[m]), ww[m])
            else:
                np.add.at(acc, (rr[m], cc[m]), scale[m] * ww[m])

    # 1) accumulate valid contributions (value*weight) and weights
    def add_valid(rr, cc, ww):
        m = valid & (rr >= 0) & (rr < rows) & (cc >= 0) & (cc < cols) & (ww > 0)
        if np.any(m):
            np.add.at(out, (rr[m], cc[m]), values[m].astype(np.float64) * ww[m])
            np.add.at(wts, (rr[m], cc[m]), ww[m])

    add_valid(r0,     c0,     w00)
    add_valid(r0+1,   c0,     w10)
    add_valid(r0,     c0+1,   w01)
    add_valid(r0+1,   c0+1,   w11)

    # 2) visit mask (any sample)
    add(visit, r0,     c0,     w00)
    add(visit, r0+1,   c0,     w10)
    add(visit, r0,     c0+1,   w01)
    add(visit, r0+1,   c0+1,   w11)

    # 3) nodata influence mask
    add(vnod, r0,     c0,     w00, mask=nod)
    add(vnod, r0+1,   c0,     w10, mask=nod)
    add(vnod, r0,     c0+1,   w01, mask=nod)
    add(vnod, r0+1,   c0+1,   w11, mask=nod)

    out_f32 = np.full((rows, cols), nodata, dtype=np.float32)
    nz = wts > 0
    out_f32[nz] = (out[nz] / wts[nz]).astype(np.float32)
    return out_f32, wts, visit, vnod

def _fill_geometric_holes_in_place(ortho, wts, visit, vnod, max_fill_radius_px=1):
    """
    Fill ONLY geometric holes (inside swath, not influenced by nodata).
    max_fill_radius_px=1 -> strictly 8-neighbor; None -> unlimited.
    """
    holes_geom = (wts == 0) & (visit > 0) & (vnod == 0)
    if not np.any(holes_geom):
        return
    if not _HAS_SCIPY:
        # SciPy not available -> leave geometric holes as-is
        return
    # nearest filled neighbor indices
    filled_mask = np.isfinite(ortho)
    dist, (ri, ci) = distance_transform_edt(~filled_mask, return_indices=True)
    if max_fill_radius_px is not None:
        # Euclidean threshold: 8-neighbor is <= sqrt(2)
        from math import sqrt
        thresh = sqrt(2) if max_fill_radius_px == 1 else float(max_fill_radius_px)
        ok = holes_geom & (dist <= thresh)
    else:
        ok = holes_geom
    ortho[ok] = ortho[ri[ok], ci[ok]]

def orthorectify_band_to_planet_grid(
    lon2d: np.ndarray,
    lat2d: np.ndarray,
    band2d: np.ndarray,
    grid: OrthoGrid
) -> np.ndarray:
    """
    Orthorectify ONE band to Planet's ortho grid using bilinear splatting
    + strict 8-neighbor fill for purely geometric gaps. True nodata preserved.
    """
    x, y = _transform_lonlat(lon2d, lat2d, grid.epsg)
    ortho, wts, visit, vnod = _splat_bilinear(band2d, x, y, grid, nodata=np.nan)
    _fill_geometric_holes_in_place(ortho, wts, visit, vnod, max_fill_radius_px=1)
    return ortho
