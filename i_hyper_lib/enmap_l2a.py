#!/usr/bin/env python3
import sys, os, xml.etree.ElementTree as ET, rasterio, grass.script as gs
from grass.pygrass.modules import Module
import contextlib

COMPOSITES = {
    "RGB": [660, 572, 478],
    "CIR": [848, 660, 572],
    "SWIR": [848, 1653, 660]
}

@contextlib.contextmanager
def suppress_stderr():
    fd, old = sys.stderr.fileno(), os.dup(sys.stderr.fileno())
    with open(os.devnull, 'w') as null:
        os.dup2(null.fileno(), fd)
    try: yield
    finally: os.dup2(old, fd); os.close(old)

def parse_band_metadata(meta_xml_path, total_bands):
    tree = ET.parse(meta_xml_path)
    root = tree.getroot()
    band_data, expected = {}, set()

    for band in root.findall(".//bandCharacterisation/bandID"):
        idx = int(band.attrib["number"])
        band_data[idx] = {
            "wavelength": float(band.findtext("wavelengthCenterOfBand")),
            "fwhm": float(band.findtext("FWHMOfBand"))
        }

    for path in [".//vnirProductQuality/expectedChannelsList", ".//swirProductQuality/expectedChannelsList"]:
        node = root.find(path)
        if node is not None and node.text:
            expected.update(int(x.strip()) for x in node.text.split(","))

    for b in range(1, total_bands + 1):
        band_data.setdefault(b, {})["valid"] = 1 if b in expected else 0
    return band_data

def find_nearest_band(wavelength, wavelengths):
    return min(range(len(wavelengths)), key=lambda i: abs(wavelengths[i] - wavelength)) + 1

def import_enmap(folder, output, composites=None):
    tif_path = os.path.join(folder, next(f for f in os.listdir(folder) if f.endswith("SPECTRAL_IMAGE.TIF")))
    meta_path = os.path.join(folder, next(f for f in os.listdir(folder) if f.endswith("METADATA.XML")))

    with rasterio.open(tif_path) as src:
        total_bands, band_meta = src.count, parse_band_metadata(meta_path, src.count)
        wavelengths, band_names = [], []

        for b in range(1, total_bands + 1):
            bname = f"{output}_b{b:03d}"
            with suppress_stderr():
                Module("r.external", input=tif_path, output=bname, band=b, flags="o", quiet=True, overwrite=True)
            wavelengths.append(band_meta[b]["wavelength"])
            band_names.append(bname)
            Module("r.colors", map=bname, color="grey", quiet=True)

        Module("i.group", group=f"{output}_group", input=band_names, quiet=True)
        Module("r.to.rast3", input=band_names, output=output, quiet=True, overwrite=True)
        Module("r3.mapcalc", expression=f"{output}_scaled = {output} / 10000.0", quiet=True, overwrite=True)
        Module("g.remove", type="raster_3d", name=output, flags="f", quiet=True)
        Module("g.rename", raster_3d=(f"{output}_scaled", output), quiet=True)

        for idx, bname in enumerate(band_names, start=1):
            wl, meta = wavelengths[idx-1], band_meta[idx]
            Module("r.support", map=bname, title=f"Band {idx}", units="nm",
                   source1=f"Wavelength: {wl} nm", source2=f"FWHM: {meta['fwhm']} nm",
                   description=f"valid: {meta['valid']}", quiet=True)

        used_bands = set()
        if composites:
            for comp in composites:
                bands = [find_nearest_band(wl, wavelengths) for wl in COMPOSITES[comp]]
                rgb_maps = [band_names[b-1] for b in bands]
                used_bands.update(rgb_maps)

                Module("r.composite",
                       red=rgb_maps[0], green=rgb_maps[1], blue=rgb_maps[2],
                       output=f"{output}_{comp.lower()}", quiet=True, overwrite=True)

                gs.info(f"Generated composite raster: {output}_{comp.lower()}")

        unused_bands = set(band_names) - used_bands
        if unused_bands:
            Module("g.remove", type="raster", name=list(unused_bands), flags="f", quiet=True)

        desc = ["Hyperspectral Metadata:", f"Bands: {total_bands}"]
        for i, wl in enumerate(wavelengths):
            meta = band_meta[i + 1]
            desc.append(f"Band: {i+1}, Wavelength: {wl}, FWHM: {meta['fwhm']}, Valid: {meta['valid']}, Unit: nm")

        Module("r3.support",
               map=output,
               title="EnMAP L2A Hyperspectral Data",
               description="\n".join(desc),
               vunit="nm",
               quiet=True)

def run_import(options):
    import_enmap(options["input"], options["output"],
                 composites=options["composites"].upper().split(",") if options.get("composites") else None)