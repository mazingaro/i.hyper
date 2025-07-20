import wx

from wxplot.profile import ProfileFrame
from mapwin.analysis import ProfileController
from core.giface import StandaloneGrassInterface
import grass.script as gs

class HyperspectralProfileFrame(ProfileFrame):
    def __init__(self, parent=None, title="Hyperspectral Raster Profile", map=None):
        self._giface = StandaloneGrassInterface()
        
        # Use current region or a default raster if none provided
        raster_map = map if map else self._get_default_raster()
        rasterList = [raster_map] if raster_map else []

        # Controller handles interaction with the map window (can be mocked for standalone)
        self.profileController = ProfileController(
            giface=self._giface,
            mapWindow=None  #TODO: get current map display?
        )

        super().__init__(
            parent=parent,
            giface=self._giface,
            rasterList=rasterList,
            units="meters",  # or derive from projection if needed
            controller=self.profileController
        )
        self.SetTitle(title)

    def _get_default_raster(self):
        """Try to retrieve a raster from the current region"""
        try:
            rasters = gs.list_strings(type="raster")
            return rasters[0] if rasters else None
        except Exception as e:
            gs.warning(f"Could not determine default raster: {e}")
            return None
