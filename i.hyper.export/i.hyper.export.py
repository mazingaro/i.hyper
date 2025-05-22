#!/usr/bin/env python

##############################################################################
# MODULE:    i.hyper.export
#
# AUTHOR(S): Alen Mangafić <alen.mangafic@gis.si>
#
# PURPOSE:   Export hyperspectral data cubes.
#
# COPYRIGHT: (C) 2025 by Alen Mangafić and the GRASS Development Team
#
#            This program is free software under the GNU General Public
#            License (>=v2). Read the file COPYING that comes with GRASS
#            for details.
##############################################################################

"""Export hyperspectral data cubes."""

# %module
# % description: Export hyperspectral data cubes.
# % keyword: raster
# % keyword: algebra
# % keyword: random
# %end
# %option G_OPT_R3_INPUT
# %end
# %option G_OPT_F_OUTPUT
# %end

import sys
import atexit
import grass.script as gs
import subprocess
import os


def clean(name):
    gs.run_command("g.remove", type="raster", name=name, flags="f", superquiet=True)


def main():
    # get input options
    options, flags = gs.parser()
    input_raster = options["input"]
    output_raster = options["output"]

    # Step 1: Decompose the 3D raster into individual 2D rasters
    temp_rasters = []
    info = gs.read_command("r3.info", map=input_raster, flags="g").strip()

    # Extract the number of layers (bands) from the info output
    num_bands_line = [line for line in info.split("\n") if "depth" in line]
    num_bands = int(num_bands_line[0].split("=")[1].strip()) if num_bands_line else 0

    for band in range(1, num_bands + 1):
        temp_raster = f"band_{band}"
        # Correct usage of r3.to.rast without the band= parameter
        gs.run_command("r3.to.rast", input=input_raster, output=temp_raster, flags="r")
        temp_rasters.append(temp_raster)

    # Step 2: Group the 2D rasters into a single 2D raster stack
    group_name = "hyperspectral_group"
    gs.run_command("i.group", group=group_name, input=",".join(temp_rasters))

    # Step 3: Export the stack as GeoTIFF
    gs.run_command("r.out.gdal", input=group_name, output=output_raster, format="GTiff", overwrite=True)

    # Clean up temporary rasters and group
    for temp_raster in temp_rasters:
        gs.run_command("g.remove", type="raster", name=temp_raster, flags="f", superquiet=True)
    gs.run_command("g.remove", type="group", name=group_name, flags="f", superquiet=True)

    # Save history into the output raster
    gs.raster_history(output_raster, overwrite=True)


if __name__ == "__main__":
    sys.exit(main())
