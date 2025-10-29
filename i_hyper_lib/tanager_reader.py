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
