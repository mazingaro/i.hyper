## DESCRIPTION

The purpose of *i.hyper.import* is to import hyperspectral imagery nto a GRASS GIS 3D raster format (`raster_3d`). It converts spectral image bands into an internally grouped 3D cube and attaches relevant per-band metadata: **wavelength**, **FWHM**, **valid** and **unit**.
Currently only **EnMAP L2A** products—i

Parameter **input** is a directory path pointing to an uncompressed EnMAP L2A product folder, which must contain both the `SPECTRAL_IMAGE.TIF` and the corresponding `METADATA.XML`.  
Parameter **output** defines the base name for the output raster_3d map and all temporary bands.

The module parses metadata, checks pixel values, finds unusable bands, and attaches all relevant spectral metadata into the final output cube.

## NOTES

- Bands that are fully zero, fully null, or contain any negative values are marked with `valid: 0`. All others are marked `valid: 1`. This is used in the other modules to mask the invalid bands.
- GDAL stderr messages about missing statistics (`Failed to compute min/max`) are suppressed internally to avoid misleading user warnings.
- The resulted hyperspectral 3D raster map has reflectance values from 0 to 1.
- All temporary rasters are cleaned up automatically after conversion to `raster_3d`.

This module is designed to be extensible—support for other sensors (e.g., PRISMA `.he5`) will be added in the future.

## EXAMPLE

```sh
i.hyper.import input=/data/ output=enmap product="EnMAP L2A"
```

## REFERENCES

- EnMAP Data & Access, GFZ German Research Centre for Geosciences. 
[EnMAP Data & Access](https://www.enmap.org/data_access/)  

- GRASS GIS Programmer’s Manual.

## SEE ALSO

- [r.external](https://grass.osgeo.org/grass-stable/manuals/r.external.html)  
- [r.to.rast3](https://grass.osgeo.org/grass-stable/manuals/r.to.rast3.html)  
- [r3.support](https://grass.osgeo.org/grass-stable/manuals/r3.support.html)  
- [i.group](https://grass.osgeo.org/grass-stable/manuals/i.group.html)

## AUTHORS

Alen Mangafić  
Geodetic Institute of Slovenia
