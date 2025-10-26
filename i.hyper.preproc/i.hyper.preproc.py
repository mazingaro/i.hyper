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
# % description: Polynomial order for Savitzky-Golay filter (0 = skip Savitzky-Golay)
# % required: no
# % answer: 0
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
# % key: b
# % description: Apply baseline correction
# % guisection: Shape/background correction
# %end

# %flag
# % key: c
# % description: Apply continuum removal: if baseline correction checked it will go directly as input
# % guisection: Shape/background correction
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
import grass.script as gs
import grass.script.array as garray
from grass.script.utils import get_lib_path
import importlib.util
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

def _import_from_i_hyper_lib(module_name):
    path = get_lib_path(modname="i_hyper_lib", libname=module_name)
    if not path:
        gs.fatal(f"Library path for {module_name} not found.")
    if path not in sys.path:
        sys.path.append(path)
    spec = importlib.util.find_spec(module_name)
    if not spec:
        gs.fatal(f"Module {module_name} not found at {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_savgol = _import_from_i_hyper_lib("sav_gol")
_basecorr = _import_from_i_hyper_lib("base_corr")
_contrem = _import_from_i_hyper_lib("continuum_rem")

_savgol_preserve_nan = _savgol._savgol_preserve_nan
_baseline_correction = _basecorr._baseline_correction
_continuum_removal = _contrem._continuum_removal


def _fill_nans_1d(x):
    v = np.asarray(x, dtype=np.float32)
    m = np.isfinite(v)
    if m.sum() < 2:
        return v
    xi = np.arange(v.size, dtype=np.float32)
    f = interp1d(xi[m], v[m], kind="linear", fill_value="extrapolate", assume_sorted=True)
    return f(xi).astype(np.float32)


def _copy_r3_metadata(src, dst):
    fd, tmp = tempfile.mkstemp(prefix="r3hist_", suffix=".txt")
    os.close(fd)
    with contextlib.suppress(FileNotFoundError):
        os.remove(tmp)
    try:
        gs.run_command("r3.support", map=src, savehistory=tmp, overwrite=True, quiet=True)
        gs.run_command("r3.support", map=dst, loadhistory=tmp, overwrite=True, quiet=True)
        gi = gs.parse_command("r3.info", flags="g", map=src)
        title = gi.get("title")
        vunit = gi.get("vertical_unit")
        if title:
            gs.run_command("r3.support", map=dst, title=title, quiet=True)
        if vunit:
            gs.run_command("r3.support", map=dst, vunit=vunit, quiet=True)
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

    # Abort immediately if no processing options are requested
def preprocess_hyperspectral(inp, out, window_length=11, polyorder=0,
                             derivative_order=0, interpolate_nodata=False,
                             clamp_negative=False, baseline=False,
                             continuum=False):

    # Explicitly abort when no operation is requested
    if (int(polyorder) == 0 and not baseline and not continuum
            and not clamp_negative and not interpolate_nodata):
        gs.fatal("No processing option selected. Use Savitzky–Golay, baseline, continuum, or value-handling flags.")

    if int(window_length) % 2 == 0:
        gs.fatal("Window length must be an odd number")

    gs.use_temp_region()
    gs.run_command("g.region", raster_3d=inp, quiet=True)

    arr_in = garray.array3d(mapname=inp, null="nan", dtype=np.float32)
    depth, rows, cols = arr_in.shape
    exterior_mask = ~np.any(np.isfinite(arr_in), axis=0)
    flat = arr_in.reshape(depth, -1).T

    if clamp_negative:
        flat = np.where(flat < 0, 0, flat).astype(np.float32)

    flat_filt = flat
    if polyorder > 0:
        flat_filt = np.apply_along_axis(
            _savgol_preserve_nan, 1, flat,
            window_length, polyorder, derivative_order, interpolate_nodata
        ).astype(np.float32)

    if baseline:
        flat_filt = np.apply_along_axis(_baseline_correction, 1, flat_filt).astype(np.float32)

    if continuum:
        flat_filt = np.apply_along_axis(_continuum_removal, 1, flat_filt).astype(np.float32)

    arr_out = flat_filt.T.reshape(depth, rows, cols)
    arr_out[:, exterior_mask] = np.nan

    out_arr = garray.array3d(dtype=np.float32)
    out_arr[...] = arr_out
    out_arr.write(mapname=out, null="nan", overwrite=True)

    _copy_r3_metadata(inp, out)
    gs.run_command("g.region", raster_3d=out, quiet=True)


def main():
    options, flags = gs.parser()

    wl = int(options["window_length"])
    poly = int(options["polyorder"])
    deriv = int(options["derivative_order"])

    baseline = bool(flags.get("b"))
    continuum = bool(flags.get("c"))
    interp = bool(flags.get("q"))
    clamp = bool(flags.get("z"))

    preprocess_hyperspectral(
        inp=options["input"],
        out=options["output"],
        window_length=wl,
        polyorder=poly,
        derivative_order=deriv,
        interpolate_nodata=interp,
        clamp_negative=clamp,
        baseline=baseline,
        continuum=continuum,
    )

if __name__ == "__main__":
    sys.exit(main())
