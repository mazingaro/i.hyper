#!/usr/bin/env python3

##############################################################################
# MODULE:    i.hyper.explore
# AUTHOR(S): Alen Mangafic <alen.mangafic@gis.si>
# PURPOSE:   Visualize hyperspectral 3D raster cube as RGB, CIR, or SWIR composite.
# COPYRIGHT: (C) 2025 by Alen Mangafic and the GRASS Development Team
##############################################################################

# %module
# % description: Visualize 3D hyperspectral cube as RGB, CIR, or SWIR composite
# % keyword: visualization
# % keyword: hyperspectral
# % keyword: 3D raster
# %end

# %option G_OPT_R3_INPUT
# % key: input
# % description: Input 3D hyperspectral cube
# % required: yes
# % guisection: Input
# %end

# %option
# % key: composites
# % type: string
# % required: yes
# % multiple: yes
# % options: RGB,CIR,SWIR
# % description: Composite types to generate
# % guisection: Settings
# %end

# %option
# % key: prefix
# % type: string
# % required: yes
# % description: Prefix for output composites
# % guisection: Settings
# %end

import sys
import grass.script as gs
from grass.pygrass.modules import Module

COMPOSITES = {
    "RGB": [660, 550, 470],
    "CIR": [850, 660, 550],
    "SWIR": [1650, 850, 660]
}

def find_nearest_band(wavelength, band_meta):
    """Find band with nearest wavelength"""
    return min(band_meta, key=lambda b: abs(band_meta[b]["Wavelength"] - wavelength))

def parse_band_metadata(r3map):
    """Extract band metadata from r3.info"""
    info = gs.read_command("r3.info", map=r3map, flags="h").splitlines()
    band_meta = {}
    in_comments = False

    for line in info:
        line = line.strip()
        if line.startswith("Comments:"):
            in_comments = True
            continue
        if not in_comments or not line.startswith("Band:"):
            continue

        try:
            parts = [p.split(":")[1].strip() for p in line.split(",")[:4]]
            band = int(parts[0])
            band_meta[band] = {
                "Wavelength": float(parts[1]),
                "Valid": int(parts[3])
            }
        except Exception as e:
            gs.warning(f"Skipping malformed line: {line}\nError: {e}")

    if not band_meta:
        gs.fatal("No valid band metadata found")
    return band_meta

def main():
    options, flags = gs.parser()
    r3map = options["input"]
    prefix = options["prefix"]
    composites = options["composites"].upper().split(",")

    band_meta = parse_band_metadata(r3map)

    for comp in composites:
        if comp not in COMPOSITES:
            gs.fatal(f"Invalid composite: {comp}")

        # Find bands
        bands = [find_nearest_band(wl, band_meta) for wl in COMPOSITES[comp]]
        gs.message(f"Using bands {bands} for {comp} composite")

        # Extract layers
        temp_maps = []
        for idx, band in enumerate(bands):
            outname = f"{prefix}_temp_{comp}_{idx+1}"
            single_cube = f"{outname}_cube"

            # Extract single band as 3D map
            gs.run_command("r3.extract",
                           input=r3map,
                           output=single_cube,
                           z=band)

            # Convert to 2D raster
            gs.run_command("r3.to.rast",
                           input=single_cube,
                           output=outname)

            # Apply grayscale color table
            Module("r.colors", map=outname, color="grey", quiet=True)

            # Clean up 3D cube
            Module("g.remove", type="raster_3d", name=single_cube, flags="f", quiet=True)

            temp_maps.append(outname)

        # Create composite
        Module("r.composite",
               red=temp_maps[0],
               green=temp_maps[1],
               blue=temp_maps[2],
               output=f"{prefix}_{comp.lower()}",
               overwrite=True)

        # Cleanup temporary 2D rasters
        Module("g.remove", type="raster", name=temp_maps, flags="f", quiet=True)

if __name__ == "__main__":
    main()
