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

"""Hyperspectral imagery preprocessing."""

# %module
# % description: Hyperspectral imagery preprocessing.
# % keyword: raster
# % keyword: algebra
# % keyword: random
# %end

# %option G_OPT_R3_INPUT
# % key: input
# % description: Input hyperspectral raster map
# % required : yes
# %end

# %option G_OPT_R3_OUTPUT
# % key: output
# % description: Output raster map after preprocessing
# % required : yes
# %end

# %option
# % key: polyorder
# % description: Polynomial order for Savitzky-Golay filter (default: 2)
# % type: integer
# % required: no
# % answer: 2
# % guisection: Savitzky-Golay
# %end

# %option
# % key: derivative_order
# % description: Derivative order for Savitzky-Golay filter (0 = just smoothing)
# % type: integer
# % required: no
# % answer: 0
# % guisection: Savitzky-Golay
# %end

# %option
# % key: window_length
# % description: Window length for Savitzky-Golay filter (odd number, default: 11)
# % type: integer
# % required: no
# % answer: 11
# % guisection: Savitzky-Golay
# %end

# %flag
# % key: i
# % description: Don't interpolate no data pixels
# % guisection: Savitzky-Golay
# %end

import sys
import grass.script as gs
import grass.script.array as garray
import numpy as np
from scipy.signal import savgol_filter

def savitzky_golay_filter(spectrum, window_length, polyorder, derivative_order=0, interpolate_nodata=True):
    window_length = int(window_length)
    polyorder = int(polyorder)
    derivative_order = int(derivative_order)
    spectrum = np.asarray(spectrum, dtype=np.float32)
    if interpolate_nodata:
        spectrum = np.nan_to_num(spectrum, nan=0.0)
    return savgol_filter(spectrum, window_length, polyorder, deriv=derivative_order, mode='interp')

def preprocess_hyperspectral(input_raster, output_raster, window_length=11, polyorder=2, derivative_order=0, interpolate_nodata=True):
    gs.run_command("g.region", raster_3d=input_raster)
    input_array = garray.array3d(input_raster)
    depth, rows, cols = input_array.shape

    output_array = garray.array3d()
    output_array[...] = np.empty((depth, rows, cols), dtype=np.float32)

    for y in range(rows):
        for x in range(cols):
            spectrum = input_array[:, y, x]
            if spectrum.ndim == 1:
                filtered_spectrum = savitzky_golay_filter(
                    spectrum, window_length, polyorder, derivative_order, interpolate_nodata
                )
                output_array[:, y, x] = filtered_spectrum

    output_array.write(output_raster, overwrite=True)

    # Set all zeros to null using r3.null
    gs.run_command("r3.null", map=output_raster, setnull=0)

def main():
    options, flags = gs.parser()
    input_raster = options["input"]
    output_raster = options["output"]
    polyorder = int(options["polyorder"])
    derivative_order = int(options["derivative_order"])
    window_length = int(options["window_length"])
    interpolate_nodata = "-i" not in flags

    preprocess_hyperspectral(
        input_raster,
        output_raster,
        window_length,
        polyorder,
        derivative_order,
        interpolate_nodata,
    )

if __name__ == "__main__":
    sys.exit(main())
