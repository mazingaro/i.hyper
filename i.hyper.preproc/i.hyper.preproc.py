#!/usr/bin/env python

##############################################################################
# MODULE:    i.hyper.preproc
#
# AUTHOR(S): Alen Mangafic <alen.mangafic@gis.si>
#
# PURPOSE:   Hyperspectral imagery preprocessing.
#
# COPYRIGHT: (C) 2025 by Alen Mangafic and the GRASS Development Team
#
##############################################################################

"""Hyperspectral imagery preprocessing"""

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
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
import grass.script as gs
import grass.script.array as garray

def fill_nans(spectrum):
    """Interpolate NaN values using linear interpolation"""
    valid = np.isfinite(spectrum)
    if np.sum(valid) < 2:
        return spectrum
    x = np.arange(len(spectrum))
    interp = interp1d(x[valid], spectrum[valid],
                      kind='linear', fill_value='extrapolate')
    return interp(x)

def savitzky_golay_filter(spectrum, window_length, polyorder,
                          derivative_order=0, interpolate_nodata=False):
    """Apply Savitzky-Golay filter with proper NaN handling"""
    spectrum = np.asarray(spectrum, dtype=np.float32)

    # Clamp negative values to zero before interpolation
    spectrum = np.where(spectrum < 0, 0, spectrum)

    # Interpolate missing values if requested
    if interpolate_nodata:
        spectrum = fill_nans(spectrum)

    if np.any(np.isnan(spectrum)):
        return np.full_like(spectrum, np.nan)

    try:
        result = savgol_filter(spectrum, window_length, polyorder,
                              deriv=derivative_order, mode='interp')
        return result
    except ValueError as e:
        gs.fatal(f"Savitzky-Golay error: {str(e)}")

def copy_r3_metadata(input_raster, output_raster):
    """Copy r3.info metadata (title, vunit, description, comments) from input to output raster_3d."""
    info = gs.read_command("r3.info", map=input_raster)
    title = None
    vunit = None
    description = []
    comments = []
    capture_comment = False

    for line in info.splitlines():
        if line.startswith("| Title:"):
            title = line.split(":", 1)[1].strip()
        elif line.startswith("| Vertical unit:"):
            vunit = line.split(":", 1)[1].strip()
        elif line.startswith("|    Data Description:"):
            capture_comment = False
        elif line.startswith("|    Comments:"):
            capture_comment = True
            continue
        elif line.startswith("|") and capture_comment:
            comment = line[1:].strip()
            if comment:
                comments.append(comment)
        elif line.startswith("|") and not capture_comment:
            description.append(line[1:].strip())

    description_text = "\n".join(description).strip()
    comments_text = "\n".join(comments).strip()

    gs.run_command("r3.support",
        map=output_raster,
        title=title if title else "",
        vunit=vunit if vunit else "",
        description=description_text,
        comments=comments_text,
        quiet=True
    )

def preprocess_hyperspectral(input_raster, output_raster, window_length=11,
                             polyorder=2, derivative_order=0,
                             interpolate_nodata=False):
    if window_length % 2 == 0:
        gs.fatal("Window length must be an odd number")

    gs.use_temp_region()
    gs.run_command("g.region", raster_3d=input_raster)

    input_array = garray.array3d(input_raster)
    depth, rows, cols = input_array.shape

    # Filter out null bands (all-NaN)
    valid_bands = ~np.all(np.isnan(input_array), axis=(1, 2))
    input_data = input_array[valid_bands, :, :]

    if input_data.shape[0] == 0:
        gs.fatal("No valid bands remaining after filtering")

    # Reshape to spectra-by-pixels
    spectra = input_data.reshape(input_data.shape[0], -1).T

    filtered = np.apply_along_axis(
        savitzky_golay_filter, 1, spectra,
        window_length, polyorder,
        derivative_order, interpolate_nodata
    )

    filtered_3d = filtered.T.reshape(input_data.shape[0], rows, cols)

    output_array = garray.array3d()
    output_array[...] = filtered_3d
    output_array.write(mapname=output_raster, overwrite=True)
    copy_r3_metadata(input_raster, output_raster)
    gs.run_command("g.region", raster_3d=output_raster)

def main():
    options, flags = gs.parser()
    window_length = int(options["window_length"])

    if window_length % 2 == 0:
        gs.fatal("Window length must be an odd number")

    preprocess_hyperspectral(
        input_raster=options["input"],
        output_raster=options["output"],
        window_length=window_length,
        polyorder=int(options["polyorder"]),
        derivative_order=int(options["derivative_order"]),
        interpolate_nodata="q" in flags
    )

if __name__ == "__main__":
    sys.exit(main())
