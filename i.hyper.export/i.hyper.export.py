#!/usr/bin/env python

##############################################################################
# MODULE:    i.hyper.export
# AUTHOR(S): Alen Mangafic <alen.mangafic@gis.si>
# PURPOSE:   Export 3D hyperspectral raster cube as a compressed multi-band GeoTIFF.
# COPYRIGHT: (C) 2025 by Alen Mangafic and the GRASS Development Team
##############################################################################

# %module
# % description: Export 3D hyperspectral raster cube to compressed multi-band GeoTIFF.
# % keyword: raster3d
# % keyword: export
# %end

# %option G_OPT_R3_INPUT
# % required: yes
# % description: Input 3D raster map
# % guisection: Input
# %end

# %option G_OPT_F_OUTPUT
# % required: yes
# % description: Output file name (GeoTIFF will be created)
# % guisection: Output
# %end

import sys
import grass.script as gs


def main():
    options, flags = gs.parser()
    input_3d = options["input"]
    output_file = options["output"]

    # Get number of slices (depth)
    info = gs.read_command("r3.info", map=input_3d, flags="g")
    num_bands = int([l for l in info.splitlines() if l.startswith("depth=")][0].split("=")[1])

    temp_rasters = []
    for i in range(1, num_bands + 1):
        name = f"{input_3d}_b{i:03d}"
        gs.run_command("r3.to.rast", input=input_3d, output=name, slice=i, quiet=True)
        temp_rasters.append(name)

    # Create a temporary group
    group_name = f"{input_3d}_export_group"
    gs.run_command("i.group", group=group_name, input=",".join(temp_rasters), quiet=True)

    # Align region to first band
    gs.run_command("g.region", raster=temp_rasters[0], align=temp_rasters[0], quiet=True)

    # Export the group
    gs.run_command("r.out.gdal",
                   input=group_name,
                   output=output_file,
                   format="GTiff",
                   type="Float32",
                   createopt="COMPRESS=DEFLATE,PREDICTOR=3,BIGTIFF=YES,INTERLEAVE=BAND",
                   overwrite=True,
                   quiet=True)

    # Cleanup
    gs.run_command("g.remove", type="raster", name=temp_rasters, flags="f", quiet=True)
    gs.run_command("g.remove", type="group", name=group_name, flags="f", quiet=True)

    gs.message(f"Exported {input_3d} to {output_file}")


if __name__ == "__main__":
    sys.exit(main())
