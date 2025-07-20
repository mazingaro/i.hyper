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
# % key: map
# % description: Input 3D raster map with hyperspectral data
# % required: yes
# %end

import sys
import grass.script as gs

def main():
    options, flags = gs.parser()
    
    try:
        # Set up GRASS GUI paths
        from grass.script.setup import set_gui_path
        set_gui_path()

        import wx
        from hyperspectral_profile_frame import HyperspectralProfileFrame
        
        app = wx.App(False)
        frame = HyperspectralProfileFrame(map=options['map'])
        frame.Show()
        app.MainLoop()
        
    except Exception as e:
        gs.fatal(f"Failed to launch GUI: {e}")
        
if __name__ == "__main__":
    main()
