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

# %option
# % key: dr_method
# % type: string
# % options: PCA,KPCA,Nystroem
# % description: Select dimensionality reduction method
# % required: no
# % guisection: Dimensionality reduction
# %end

# %option
# % key: dr_components
# % type: integer
# % description: Number of components to retain
# % required: no
# % answer: 0
# % guisection: Dimensionality reduction
# %end

# %option
# % key: dr_kernel
# % type: string
# % description: Kernel type (used only for KPCA or Nystroem)
# % options: linear,rbf,poly,sigmoid
# % required: no
# % answer: rbf
# % guisection: Dimensionality reduction
# %end

# %option
# % key: dr_gamma
# % type: double
# % description: Kernel gamma (KPCA/Nystroem only)
# % required: no
# % answer: 0.01
# % guisection: Dimensionality reduction
# %end

# %option
# % key: dr_degree
# % type: integer
# % description: Polynomial degree (only used if kernel=poly)
# % required: no
# % answer: 3
# % guisection: Dimensionality reduction
# %end

# %option
# % key: dr_bands
# % type: string
# % description: Wavelength intervals or single values to include before dimensionality reduction (e.g., 400–700,850–1300,2200)
# % required: no
# % guisection: Dimensionality reduction
# %end

# %option G_OPT_F_OUTPUT
# % key: dr_export
# % description: Optional output file to export fitted reduction model (.pkl, for reuse)
# % required: no
# % guisection: Dimensionality reduction
# %end

# %flag
# % key: b
# % description: Apply baseline correction
# % guisection: Additional corrections
# %end

# %flag
# % key: c
# % description: Apply continuum removal
# % guisection: Additional corrections
# %end

# %flag
# % key: q
# % description: Interpolate missing values in valid bands
# % guisection: Additional corrections
# %end

# %flag
# % key: z
# % description: Clamp negative values to zero
# % guisection: Additional corrections
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
_dimred = _import_from_i_hyper_lib("dim_red")

_savgol_preserve_nan = _savgol._savgol_preserve_nan
_baseline_correction = _basecorr._baseline_correction
_continuum_removal = _contrem._continuum_removal
_apply_dimensionality_reduction = _dimred._apply_dimensionality_reduction


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


def _set_dr_metadata(outmap, method, info):
    lines = []
    if method == "PCA" and "explained_variance_ratio" in info:
        var = info["explained_variance_ratio"]
        lines.append("Principal Component Analysis (PCA)")
        for i, v in enumerate(var, 1):
            lines.append(f"Component {i}: {v*100:.2f}% variance explained")
    elif method == "KPCA":
        lines.append(f"Kernel PCA (kernel={info.get('kernel')}, gamma={info.get('gamma')}, degree={info.get('degree')})")
        lines.append(f"Components: {info.get('n_components')}")
    elif method == "Nystroem":
        lines.append(f"Nystroem approximation (kernel={info.get('kernel')}, gamma={info.get('gamma')}, degree={info.get('degree')})")
        lines.append(f"Components: {info.get('n_components')}")
    if lines:
        gs.run_command("r3.support", map=outmap, description="\n".join(lines), quiet=True)


def preprocess_hyperspectral(inp, out, window_length=11, polyorder=0,
                             derivative_order=0, interpolate_nodata=False,
                             clamp_negative=False, baseline=False,
                             continuum=False, dr_method=None,
                             dr_components=0, dr_kernel="rbf",
                             dr_gamma=0.01, dr_degree=3,
                             dr_bands=None, dr_export=None):

    if (int(polyorder) == 0 and not baseline and not continuum
            and not clamp_negative and not interpolate_nodata
            and not dr_method):
        gs.fatal("No processing option selected. Use preprocessing or dimensionality reduction parameters.")

    if int(window_length) % 2 == 0:
        gs.fatal("Window length must be an odd number")

    steps = []
    if polyorder > 0:
        steps.append("Savitzky–Golay")
    if baseline:
        steps.append("Baseline correction")
    if continuum:
        steps.append("Continuum removal")
    if dr_method:
        steps.append(dr_method)
    gs.message(" → ".join(steps) if steps else "No operations selected")

    gs.use_temp_region()
    gs.run_command("g.region", raster_3d=inp, quiet=True)

    arr_in = garray.array3d(mapname=inp, null="nan", dtype=np.float32)
    depth, rows, cols = arr_in.shape
    exterior_mask = ~np.any(np.isfinite(arr_in), axis=0)
    flat = arr_in.reshape(depth, -1).T

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

    # Interpolate missing values if requested (after all preprocessing)
    if interpolate_nodata:
        gs.message("Interpolating missing values across spectral bands...")
        for i in range(flat_filt.shape[0]):
            row = flat_filt[i, :]
            if np.isnan(row).any():
                flat_filt[i, :] = _fill_nans_1d(row)

    # Clamp negative values
    if clamp_negative:
        flat_filt = np.where(flat_filt < 0, 0, flat_filt).astype(np.float32)

    # Remove rows still containing NaNs (invalid spectra)
    nan_rows = np.isnan(flat_filt).any(axis=1)
    if nan_rows.any():
        gs.message(f"Removing {nan_rows.sum()} invalid spectra before {dr_method}...")
        flat_filt = flat_filt[~nan_rows]

    arr_out = flat_filt.T.reshape(-1, rows, cols)
    arr_out[:, exterior_mask] = np.nan

    out_arr = garray.array3d(dtype=np.float32)
    out_arr[...] = arr_out
    out_arr.write(mapname=out, null="nan", overwrite=True)

    if dr_method:
        _set_dr_metadata(out, dr_method, dr_info or {})
    else:
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
    dr_method = options["dr_method"] or None
    dr_components = int(options["dr_components"])
    dr_kernel = options["dr_kernel"]
    dr_gamma = float(options["dr_gamma"])
    dr_degree = int(options["dr_degree"])
    dr_bands = options["dr_bands"] or None
    dr_export = options["dr_export"] or None

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
        dr_method=dr_method,
        dr_components=dr_components,
        dr_kernel=dr_kernel,
        dr_gamma=dr_gamma,
        dr_degree=dr_degree,
        dr_bands=dr_bands,
        dr_export=dr_export
    )

if __name__ == "__main__":
    sys.exit(main())

