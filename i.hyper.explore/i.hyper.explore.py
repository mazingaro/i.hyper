#!/usr/bin/env python3

##############################################################################
# MODULE:    i.hyper.explore
# AUTHOR(S): Alen Mangafic <alen.mangafic@gis.si>
# PURPOSE:   Visualize spectra from hyperspectral 3D raster maps.
# COPYRIGHT: (C) 2025 by Alen Mangafic and the GRASS Development Team
##############################################################################

# %module
# % description: Visualize spectra from hyperspectral 3D raster maps.
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
# % key: prefix
# % type: string
# % required: yes
# % description: Prefix for output composites
# % guisection: Settings
# %end