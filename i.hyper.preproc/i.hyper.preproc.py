#!/usr/bin/env python

##############################################################################
# MODULE:    i.hyper.preproc
# AUTHOR(S): Alen Mangafic <alen.mangafic@gis.si>
# PURPOSE:   Hyperspectral imagery preprocessing.
# COPYRIGHT: (C) 2025 by Alen Mangafic and the GRASS Development Team
##############################################################################

# %module
# % description: General hyperspectral data preprocessing
# % keyword: raster
# % keyword: hyperspectral
# % keyword: preprocessing
# %end

# %option G_OPT_R3_INPUT
# % key: input
# % description: Input hyperspectral raster map
# % required: yes
# % guisection: Input
# %end

# %option G_OPT_R3_OUTPUT
# % key: output
# % description: Output preprocessed raster map
# % required: yes
# % guisection: Output
# %end

# %option
# % key: polyorder
# % type: integer
# % description: Polynomial order for Savitzky-Golay filter
# % required: no
# % answer: 2
# % guisection: Savitzky-Golay
# %end

# %option
# % key: derivative_order
# % type: integer
# % description: Derivative order (0 = smoothing only)
# % required: no
# % answer: 0
# % guisection: Savitzky-Golay
# %end

# %option
# % key: window_length
# % type: integer
# % description: Window length (must be odd number)
# % required: no
# % answer: 11
# % guisection: Savitzky-Golay
# %end

# %flag
# % key: q
# % description: Interpolate missing values in valid bands
# % guisection: Null/Value handling
# %end

# %flag
# % key: z
# % description: Clamp negative values to zero
# % guisection: Null/Value handling
# %end

import sys
import os
import tempfile
import contextlib
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
import grass.script as gs
import grass.script.array as garray

# linear fill inside a spectrum for internal NaNs (used only with -q)
def _fill_nans_1d(x):
    v = np.asarray(x, dtype=np.float32)
    m = np.isfinite(v)
    if m.sum() < 2:
        return v
    xi = np.arange(v.size, dtype=np.float32)
    f = interp1d(xi[m], v[m], kind="linear", fill_value="extrapolate", assume_sorted=True)
    return f(xi).astype(np.float32)

# Savitzky–Golay on one spectrum, respecting NaNs as no-data
def _savgol_preserve_nan(spec, win, poly, deriv, interp_nodata):
    s = np.asarray(spec, dtype=np.float32)
    nanmask = ~np.isfinite(s)

    if interp_nodata:
        s = _fill_nans_1d(s)
        # any leading/trailing NaNs (pure no-data pixels) stay NaN:
        s[nanmask] = np.nan

    # run filter only on finite samples; keep NaNs as NaNs
    out = np.full_like(s, np.nan, dtype=np.float32)
    valid = np.isfinite(s)
    if valid.sum() >= win:
        out[valid] = savgol_filter(s[valid], win, poly, deriv=deriv, mode="interp").astype(np.float32)
    else:
        out[valid] = s[valid]
    return out

def _copy_r3_metadata(src, dst):
    # copy history/title/description/vunit safely via a temp history file
    fd, tmp = tempfile.mkstemp(prefix="r3hist_", suffix=".txt")
    os.close(fd)
    with contextlib.suppress(FileNotFoundError):
        os.remove(tmp)
    try:
        gs.run_command("r3.support", map=src,  savehistory=tmp, overwrite=True, quiet=True)
        gs.run_command("r3.support", map=dst, loadhistory=tmp, overwrite=True, quiet=True)

        gi = gs.parse_command("r3.info", flags="g", map=src)
        title = gi.get("title")
        vunit = gi.get("vertical_unit")
        if title:
            gs.run_command("r3.support", map=dst, title=title, quiet=True)
        if vunit:
            gs.run_command("r3.support", map=dst, vunit=vunit, quiet=True)

        # copy multi-line description block (Data Description) if present
        txt = gs.read_command("r3.info", map=src)
        lines, grab = [], False
        for line in txt.splitlines():
            s = line.rstrip()
            if s.strip().startswith("|   Data Description:"):
                grab = True
                continue
            if grab:
                if s.strip().startswith("|   Comments:"):
                    break
                if s.startswith("|"):
                    content = s[1:].strip()
                    if content:
                        lines.append(content)
        if lines:
            gs.run_command("r3.support", map=dst, description="\n".join(lines), quiet=True)
    finally:
        with contextlib.suppress(Exception):
            os.remove(tmp)

def preprocess_hyperspectral(inp, out, window_length=11, polyorder=2,
                             derivative_order=0, interpolate_nodata=False,
                             clamp_negative=False):
    if window_length % 2 == 0:
        gs.fatal("Window length must be an odd number")

    gs.use_temp_region()
    gs.run_command("g.region", raster_3d=inp, quiet=True)

    # Read as float32 and treat GRASS NULL as NaN directly
    arr_in = garray.array3d(mapname=inp, null="nan", dtype=np.float32)
    depth, rows, cols = arr_in.shape

    # Exterior nodata mask (pixels that are NULL in ALL bands); keep them NULL
    exterior_mask = ~np.any(np.isfinite(arr_in), axis=0)

    flat = arr_in.reshape(depth, -1).T  # (rows*cols, depth)

    if clamp_negative:
        flat = np.where(flat < 0, 0, flat).astype(np.float32)

    flat_filt = np.apply_along_axis(
        _savgol_preserve_nan, 1, flat,
        window_length, polyorder, derivative_order, interpolate_nodata
    ).astype(np.float32)

    arr_out = flat_filt.T.reshape(depth, rows, cols)

    # Reapply exterior NaNs (ensures transparent outside footprint)
    arr_out[:, exterior_mask] = np.nan

    out_arr = garray.array3d(dtype=np.float32)
    out_arr[...] = arr_out
    # Write with null=nan so NaNs become GRASS NULL in the result
    out_arr.write(mapname=out, null="nan", overwrite=True)

    _copy_r3_metadata(inp, out)
    gs.run_command("g.region", raster_3d=out, quiet=True)

def main():
    options, flags = gs.parser()
    wl = int(options["window_length"])
    if wl % 2 == 0:
        gs.fatal("Window length must be an odd number")

    preprocess_hyperspectral(
        inp=options["input"],
        out=options["output"],
        window_length=wl,
        polyorder=int(options["polyorder"]),
        derivative_order=int(options["derivative_order"]),
        interpolate_nodata=("q" in flags),
        clamp_negative=("z" in flags),
    )

if __name__ == "__main__":
    sys.exit(main())
