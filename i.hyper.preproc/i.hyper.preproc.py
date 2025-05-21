#!/usr/bin/env python

##############################################################################
# MODULE:    i.hyper.preproc
#
# AUTHOR(S): Alen Mangafić <alen.mangafic@gis.si>
#
# PURPOSE:   Hyperspectral imagery preprocessing.
#
# COPYRIGHT: (C) 2025 by Alen Mangafić and the GRASS Development Team
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
# %option G_OPT_R_INPUT
# %end
# %option G_OPT_R_OUTPUT
# %end


import sys
import atexit
import grass.script as gs


def clean(name):
    gs.run_command("g.remove", type="raster", name=name, flags="f", superquiet=True)


def main():
    # get input options
    options, flags = gs.parser()
    input_raster = options["input"]
    output_raster = options["output"]

    # crete a temporary raster that will be removed upon exit
    temporary_raster = gs.append_node_pid("gauss")
    atexit.register(clean, temporary_raster)

    # if changing computational region is needed, uncomment
    # gs.use_temp_region()

    # verbose message with translatable string
    gs.verbose(_("Generating temporary raster {tmp}").format(tmp=temporary_raster))
    # run analysis
    gs.run_command("r.surf.gauss", output=temporary_raster)
    gs.mapcalc(f"{output_raster} = {input_raster} + {temporary_raster}")

    # save history into the output raster
    gs.raster_history(output_raster, overwrite=True)


if __name__ == "__main__":
    sys.exit(main())
