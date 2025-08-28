#!/usr/bin/env python3
import sys, os, xml.etree.ElementTree as ET, rasterio, grass.script as gs
from grass.pygrass.modules import Module
import contextlib
from statistics import mean

COMPOSITES = {
    "RGB": [660, 572, 478],
    "CIR": [848, 660, 572],
    "SWIR-agriculture": [848, 1653, 660],
    "SWIR-geology": [2200, 848, 572]
}

@contextlib.contextmanager
def suppress_stderr():
    fd, old = sys.stderr.fileno(), os.dup(sys.stderr.fileno())
    with open(os.devnull, 'w') as null:
        os.dup2(null.fileno(), fd)
    try:
        yield
    finally:
        os.dup2(old, fd)
        os.close(old)

def parse_band_metadata(meta_xml_path, tif_path, total_bands):
    tree = ET.parse(meta_xml_path)
    root = tree.getroot()
    band_data, expected = {}, set()
    for band in root.findall(".//bandCharacterisation/bandID"):
        idx = int(band.attrib["number"])
        band_data[idx] = {
            "wavelength": float(band.findtext("wavelengthCenterOfBand")),
            "fwhm": float(band.findtext("FWHMOfBand")),
            "valid": 0
        }
    for path in [".//vnirProductQuality/expectedChannelsList",
                 ".//swirProductQuality/expectedChannelsList"]:
        node = root.find(path)
        if node is not None and node.text:
            expected.update(int(x.strip()) for x in node.text.split(","))
    with rasterio.open(tif_path) as src:
        for b in range(1, total_bands + 1):
            stats_valid = src.tags(b).get('STATISTICS_VALID_PERCENT')
            valid = 1 if (b in expected and stats_valid and float(stats_valid) > 0) else 0
            if b not in band_data:
                band_data[b] = {"valid": valid}
            else:
                band_data[b]["valid"] = valid
    return band_data

def find_nearest_band(wavelength, wavelengths):
    return min(range(len(wavelengths)), key=lambda i: abs(wavelengths[i] - wavelength)) + 1

def import_enmap(folder, output, composites=None, custom_wavelengths=None, strength_val=96):
    tif_path = os.path.join(folder, next(f for f in os.listdir(folder) if f.endswith("SPECTRAL_IMAGE.TIF")))
    meta_path = os.path.join(folder, next(f for f in os.listdir(folder) if f.endswith("METADATA.XML")))
    with rasterio.open(tif_path) as src:
        total_bands = src.count
        band_meta = parse_band_metadata(meta_path, tif_path, total_bands)
        valid_bands = [b for b in range(1, total_bands + 1) if band_meta[b].get('valid', 0) == 1]
        wavelengths = []
        band_names = []
        for b in valid_bands:
            bname = f"{output}_b{b:03d}"
            with suppress_stderr():
                Module("r.external", input=tif_path, output=bname, band=b, flags="o", quiet=True, overwrite=True)
            wavelengths.append(band_meta[b]["wavelength"])
            band_names.append(bname)
            Module("r.colors", map=bname, color="grey.eq", quiet=True)

        rgb_target = COMPOSITES["RGB"]
        rgb_indices = [find_nearest_band(wl, wavelengths) for wl in rgb_target]
        rgb_enhanced = {i: band_names[i - 1] for i in rgb_indices}

        gs.use_temp_region()
        Module("g.region", raster=band_names[0], quiet=True)

        if composites:
            for comp in composites:
                if comp not in COMPOSITES:
                    continue
                bands = [find_nearest_band(wl, wavelengths) for wl in COMPOSITES[comp]]
                rgb_maps = [rgb_enhanced[b] if b in rgb_enhanced else band_names[b - 1] for b in bands]
                if comp.upper() == "RGB":
                    Module("i.colors.enhance", red=rgb_maps[0], green=rgb_maps[1], blue=rgb_maps[2],
                           strength=str(strength_val), flags="p", quiet=True)
                else:
                    Module("i.colors.enhance", red=rgb_maps[0], green=rgb_maps[1], blue=rgb_maps[2],
                           strength=str(strength_val), quiet=True)
                outname = f"{output}_{comp.lower().replace('-', '_')}"
                Module("r.composite", red=rgb_maps[0], green=rgb_maps[1], blue=rgb_maps[2],
                       output=outname, quiet=True, overwrite=True)
                gs.info(f"Generated composite raster: {outname}")

        if custom_wavelengths:
            custom_indices = [find_nearest_band(wl, wavelengths) for wl in custom_wavelengths]
            custom_maps = [rgb_enhanced[b] if b in rgb_enhanced else band_names[b - 1] for b in custom_indices]
            Module("i.colors.enhance", red=custom_maps[0], green=custom_maps[1], blue=custom_maps[2],
                   strength=str(strength_val), quiet=True)
            Module("r.composite", red=custom_maps[0], green=custom_maps[1], blue=custom_maps[2],
                   output=f"{output}_custom", quiet=True, overwrite=True)
            gs.info(f"Generated custom composite raster: {output}_custom")

        wl = wavelengths
        if len(wl) > 1:
            diffs = [wl[i+1] - wl[i] for i in range(len(wl)-1)]
            tbres_nm = float(f"{max(1e-6, mean(diffs)):.6f}")
        else:
            tbres_nm = 1.0
        bottom_nm = wl[0]
        top_nm = bottom_nm + tbres_nm * (len(band_names) - 1)

        Module("g.region", raster=band_names[0], b=bottom_nm, t=top_nm, tbres=tbres_nm, quiet=True)
        Module("r.to.rast3", input=band_names, output=output, quiet=True, overwrite=True)
        Module("r3.mapcalc", expression=f"{output}_scaled = {output} / 10000.0", quiet=True, overwrite=True)
        Module("g.remove", type="raster_3d", name=output, flags="f", quiet=True)
        Module("g.rename", raster_3d=(f"{output}_scaled", output), quiet=True)

        desc = ["Hyperspectral Metadata:", f"Valid Bands: {len(valid_bands)}"]
        for idx, b in enumerate(valid_bands, 1):
            meta = band_meta[b]
            desc.append(f"Band {idx}: {meta['wavelength']} nm, FWHM: {meta['fwhm']} nm")
            Module("r.support", map=band_names[idx - 1],
                   title=f"Band {b}", units="nm",
                   source1=f"Wavelength: {meta['wavelength']} nm",
                   source2=f"FWHM: {meta['fwhm']} nm",
                   description="Validated band", quiet=True)

        Module("r3.support", map=output,
               title="EnMAP L2A Hyperspectral Data",
               description="\n".join(desc),
               unit="reflectance",
               vunit="nm",
               quiet=True)

        gs.del_temp_region()
        Module("g.remove", type="raster", name=band_names, flags="f", quiet=True)

def run_import(options, flags):
    custom = None
    if options.get("composites_custom"):
        try:
            custom = [float(x.strip()) for x in options["composites_custom"].split(",")]
            if len(custom) != 3:
                raise ValueError
        except Exception:
            gs.fatal("Invalid format for composites_custom. Usage example: 850,1650,660")
    strength_opt = options.get("strength")
    if strength_opt is None or str(strength_opt).strip() == "":
        strength_val = 96
    else:
        try:
            strength_val = int(str(strength_opt).strip())
        except Exception:
            gs.fatal("Invalid strength. Provide an integer 0-100.")
        if not (0 <= strength_val <= 100):
            gs.fatal("Invalid strength. Provide an integer 0-100.")
    import_enmap(
        options["input"],
        options["output"],
        composites=[c.strip() for c in options["composites"].split(",")] if options.get("composites") else None,
        custom_wavelengths=custom,
        strength_val=strength_val
    )
