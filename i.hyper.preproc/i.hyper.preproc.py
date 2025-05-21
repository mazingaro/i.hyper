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
#            This program is free software under the GNU General Public
#            License (>=v2). Read the file COPYING that comes with GRASS
#            for details.
##############################################################################

"""Hyperspectral imagery preprocessing."""

# %module
# % description: Hyperspectral imagery preprocessing.
# % keyword: raster
# % keyword: algebra
# % keyword: random
# %end

# Input/Output options
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
# % key: derivative_order
# % description: Derivative order for Savitzky-Golay filter
# % type: integer
# % required: no
# % answer: 1
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
import atexit
import grass.script as gs
import numpy as np
from scipy.signal import savgol_filter
import grass.script.array as garray

def savitzky_golay_filter(spectrum, window_length, polyorder, interpolate_nodata=True):
    # Ensure window_length and polyorder are integers
    window_length = int(window_length)
    polyorder = int(polyorder)

    # Ensure the spectrum is a numeric array (float)
    spectrum = np.asarray(spectrum, dtype=np.float32)  # Convert to numeric array (float32)

    # Check if spectrum is 1D array
    if spectrum.ndim == 1:
        if interpolate_nodata:
            spectrum = np.nan_to_num(spectrum, nan=0.0)  # Replace NaN with 0 if interpolate is True
        return savgol_filter(spectrum, window_length, polyorder, deriv=1, mode='nearest')
    else:
        raise ValueError(f"Spectrum should be a 1D array, but got a {spectrum.ndim}D array.")

def preprocess_hyperspectral(input_raster, output_raster, window_length=11, polyorder=1, interpolate_nodata=True):
    # Read the 3D raster into a numpy array using garray.array3d
    input_array = garray.array3d(input_raster)

    # Apply the Savitzky-Golay filter to each pixel in the raster
    depths, rows, cols = input_array.shape
    for y in range(rows):
        for x in range(cols):
            # Extract the spectrum for this pixel (band values)
            spectrum = input_array[:, y, x]

            # Ensure spectrum is a valid 1D array before applying filter
            if spectrum.ndim == 1:  # Check if spectrum is a 1D array
                filtered_spectrum = savitzky_golay_filter(spectrum, window_length, polyorder, interpolate_nodata)
                input_array[:, y, x] = filtered_spectrum
            else:
                raise ValueError(f"Invalid spectrum shape at (y={y}, x={x}): expected 1D array, got {spectrum.shape}")

    # Write the processed array back to GRASS raster
    input_array.write(output_raster, overwrite=True)

def main():
    options, flags = gs.parser()
    input_raster = options["input"]
    output_raster = options["output"]

    derivative_order = options["derivative_order"] or 1
    window_length = options["window_length"] or 11
    interpolate_nodata = "-i" in flags

    # Preprocess hyperspectral data
    preprocess_hyperspectral(input_raster, output_raster, window_length, derivative_order, interpolate_nodata)

if __name__ == "__main__":
    sys.exit(main())
