#!/usr/bin/env python3

##############################################################################
# MODULE:    i.hyper.explore
# AUTHOR(S): Tomaz Zagar <tomaz.zagar@gis.si>
# PURPOSE:   Visualize spectra from hyperspectral 3D raster maps.
# COPYRIGHT: (C) 2025 by Tomaz Zagar and the GRASS Development Team
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

# %option G_OPT_M_COORDS
# % key: coordinates
# % description: Comma separated list of coordinates
# % multiple: yes
# % required: yes
# %end

# %flag
# % key: p
# % description: Plot spectra using matplotlib
# %end

# %option G_OPT_F_OUTPUT
# % key: output
# % required: no
# % label: Output plot file (.png, .pdf, .svg). If not set and -p is given, shows an interactive window
# %end

# %option
# % key: size
# % type: string
# % label: Size of output image as width,height in pixels (used only with output=)
# % required: no
# %end


import sys
import grass.script as gs
import json, re

def _band_count(mapname):
    info = gs.parse_command("r3.info", flags="g", map=mapname)
    d = int(info["depths"])
    if d <= 0:
        gs.fatal("Invalid band count (depths) reported by r3.info")
    return d

def _band_wavelengths(mapname, expected):
    """
    Parse wavelengths and FWHM from r3.info description/comments.
    Returns two lists (len == expected):
      - wavelengths[i] -> float or None
      - fwhm[i]        -> float or None
    """
    txt = gs.read_command("r3.info", map=mapname)
    wavelengths = [None] * expected
    fwhm = [None] * expected

    # number pattern handles ints, floats, scientific notation
    num = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
    # e.g.: "Band 157: 2369.21 nm, FWHM: 7.47001 nm"
    pat = re.compile(
        rf"Band\s+(\d+)\s*:\s*({num})\s*nm(?:,\s*FWHM:\s*({num})\s*nm)?",
        re.IGNORECASE
    )

    for line in txt.splitlines():
        line = line.strip()
        if line.startswith("|"):
            line = line.strip("| ").rstrip("| ").strip()

        m = pat.search(line)   # search anywhere in the line
        if m:
            idx = int(m.group(1))  # 1-based
            if 1 <= idx <= expected:
                wavelengths[idx - 1] = float(m.group(2))
                if m.group(3) is not None:
                    fwhm[idx - 1] = float(m.group(3))

    return wavelengths, fwhm

def _sample_all_bands_at_point(mapname, e, n, band_count, sep="|", null_marker="*"):
    """
    Calls r3.what once (2D coords) and returns list of band_count values (float or None).
    """
    out = gs.read_command(
        "r3.what",
        input=mapname,
        coordinates=f"{e},{n}",     # pass through unchanged
        separator=sep,
        null_value= null_marker,
        quiet=True,
    )
    line = out.strip().splitlines()
    if not line:
        return [None] * band_count
    cols = line[0].split(sep)
    vals_raw = cols[2:]             # first two values are coordinates
    vals = []
    for v in vals_raw:
        v = v.strip()
        if v == null_marker or v == "":
            vals.append(None)
        else:
            try:
                vals.append(float(v))
            except ValueError:
                vals.append(v)
    if len(vals) < band_count:
        vals += [None] * (band_count - len(vals))
    return vals[:band_count]

def _sample_at_3dpoint(mapname, e, n, z, sep="|", null_marker="*"):
    out = gs.read_command(
        "r3.what",
        input=mapname,
        coordinates_3d=f"{e},{n}," + str(z),
        separator=sep,
        null_value= null_marker,
        quiet=True,
    )
    
    return out.strip().split(sep)[-1]

def _plot_results(wavelengths, points, title=None, xlabel="Wavelength (nm)",
                  ylabel="Value", output=None, size=None):
    import numpy as np
    import matplotlib
    # Use a non-interactive backend if saving to file (no GUI required)
    if output:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    wl = np.array([np.nan if w is None else float(w) for w in wavelengths], dtype=float)

    fig, ax = plt.subplots()
    for p in points:
        vals = np.array([np.nan if v is None else float(v) for v in p["values"]], dtype=float)
        mask = np.isfinite(wl) & np.isfinite(vals)
        if not np.any(mask):
            continue
        ax.plot(wl[mask], vals[mask], label=f"E={p['x']:.3f}, N={p['y']:.3f}")

    if title:
        ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True, alpha=0.3)

    if size and output:
        # size="width,height" in pixels -> convert to inches at 100 dpi
        try:
            w_px, h_px = [int(s) for s in size.split(",")]
            fig.set_size_inches(w_px / 100.0, h_px / 100.0)
        except Exception:
            pass

    if output:
        fig.savefig(output, bbox_inches="tight")
    else:
        plt.show()


def main(options, flags):
    mapname = options["map"]
    coords_opt = options["coordinates"]  # list (multiple=yes)
    coords = coords_opt.split(",")

    band_count = _band_count(mapname)
    wavelengths, fwhm = _band_wavelengths(mapname, band_count)

    if len(coords) % 2 != 0:
        gs.fatal("Coordinates list must contain an even number of values (E,N pairs)")

    gs.use_temp_region()
    gs.run_command("g.region", raster_3d=mapname)

    points = []

    for i in range(0, len(coords), 2):
        try:
            e = float(coords[i])
            n = float(coords[i + 1])
        except ValueError:
            gs.fatal(f"Non-numeric coordinate at position {i}: {coords[i]}, {coords[i+1]}")

        # Single 2D r3.what call returns all bands at once
        values = _sample_all_bands_at_point(mapname, e, n, band_count)

        points.append({"x": e, "y": n, "values": values})

    results = {
        "map": mapname,
        "wavelength_nm": wavelengths,
        "points": points,
    }

    print(json.dumps(results, ensure_ascii=False))

    # Optional plotting (like g.gui.tplot pattern)
    if flags.get("p") or options.get("output"):
        title = f"Spectra @ {mapname}"
        xlabel = "Wavelength (nm)"
        ylabel = "Value"
        _plot_results(
            wavelengths=results["wavelength_nm"],
            points=results["points"],
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            output=options.get("output"),
            size=options.get("size"),
        )



if __name__ == "__main__":
    options, flags = gs.parser()
    main(options, flags)
