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
    input_3d_full = options["input"]
    output_file = options["output"]

    input_3d = input_3d_full.split("@")[0]
    base = f"{input_3d}_slice"

    # Convert 3D raster to 2D slices
    gs.run_command("r3.to.rast", input=input_3d_full, output=base, quiet=True)

    # Use g.list to get unqualified raster names (safe)
    raster_list = gs.parse_command("g.list", type="raster", pattern=f"{base}_*", flags="m")

    # Sort while stripping any mapset
    def _get_index(rname):
        r = rname.split("@")[0]
        return int(r[len(base) + 1:])
    raster_list = sorted(raster_list, key=_get_index)

    if not raster_list:
        gs.fatal(f"No valid slice maps found with base name {base}_*")

    group_name = f"{input_3d}_export_group"
    gs.run_command("i.group", group=group_name, input=",".join(raster_list), quiet=True)
    gs.run_command("g.region", raster=raster_list[0], align=raster_list[0], quiet=True)

    gs.run_command("r.out.gdal",
                   input=group_name,
                   output=output_file,
                   format="GTiff",
                   createopt="COMPRESS=DEFLATE,PREDICTOR=3,BIGTIFF=YES,INTERLEAVE=BAND",
                   overwrite=True,
                   quiet=True)

    gs.run_command("g.remove", type="raster", name=raster_list, flags="f", quiet=True)
    gs.run_command("g.remove", type="group", name=group_name, flags="f", quiet=True)

    gs.message(f"Exported {input_3d_full} to {output_file} as multi-band GeoTIFF")


if __name__ == "__main__":
    sys.exit(main())
