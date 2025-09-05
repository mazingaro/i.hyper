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
# % description: Input 3D raster map(s) with hyperspectral data (comma-separated)
# % multiple: yes
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
# % description: Print JSON to stdout instead of plotting
# %end

# %option G_OPT_F_OUTPUT
# % key: output
# % required: no
# % label: Output plot file (.png, .pdf, .svg). If not set, opens an interactive window
# %end

# %option
# % key: size
# % type: string
# % label: Size of output image as width,height in pixels (used only with output=)
# % required: no
# %end

import grass.script as gs
import json
import re

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
        m = pat.search(line)
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
        coordinates=f"{e},{n}",
        separator=sep,
        null_value=null_marker,
        quiet=True,
    )
    line = out.strip().splitlines()
    if not line:
        return [None] * band_count
    cols = line[0].split(sep)
    vals_raw = cols[2:]  # first two values are coordinates
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
        coordinates_3d=f"{e},{n},{z}",
        separator=sep,
        null_value= null_marker,
        quiet=True,
    )
    return out.strip().split(sep)[-1]

def _plot_results_multi(datasets, title=None, xlabel="Wavelength (nm)",
                        ylabel="Value", output=None, size=None):
    """
    datasets: list of dicts:
      {
        "map": <name>,
        "wavelength_nm": [...],
        "points": [ {"x": E, "y": N, "values": [...]}, ... ]
      }
    Styling:
      - Linestyle varies by MAP
      - Color varies by POINT index (consistent across maps)
    """
    import numpy as np
    import matplotlib
    if output:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    linestyles = ["-", "--", "-.", ":", (0, (5, 1)), (0, (3, 1, 1, 1)), (0, (1, 1))]
    fig, ax = plt.subplots()

    # How many points total (use max across datasets)
    num_points = max((len(ds["points"]) for ds in datasets), default=0)

    # Colors from current Matplotlib cycle
    prop_cycle = plt.rcParams.get("axes.prop_cycle", None)
    colors = (prop_cycle.by_key().get("color", []) if prop_cycle else [])
    if not colors:
        colors = ["C0","C1","C2","C3","C4","C5","C6","C7","C8","C9"]

    # Plot: loop by point index to keep color consistent across maps
    for pi in range(num_points):
        color = colors[pi % len(colors)]
        for mi, ds in enumerate(datasets):
            if pi >= len(ds["points"]):
                continue
            wl = np.asarray([np.nan if w is None else float(w)
                             for w in ds["wavelength_nm"]], dtype=float)
            p = ds["points"][pi]
            vals = np.asarray([np.nan if v is None else float(v)
                               for v in p["values"]], dtype=float)
            mask = np.isfinite(wl) & np.isfinite(vals)
            if not np.any(mask):
                continue
            ls = linestyles[mi % len(linestyles)]
            ax.plot(wl[mask], vals[mask], linestyle=ls, color=color, linewidth=1.6)

    if title:
        ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)

    # Legends: maps (linestyles) and points (colors)
    map_handles = [
        Line2D([0], [0], linestyle=linestyles[mi % len(linestyles)],
               color="black", label=ds["map"])
        for mi, ds in enumerate(datasets)
    ]

    point_handles = []
    first_points = datasets[0]["points"] if datasets and datasets[0]["points"] else []
    for pi in range(num_points):
        label = (f"P{pi+1}" if pi >= len(first_points)
                 else f"P{pi+1}: E={first_points[pi]['x']:.3f}, N={first_points[pi]['y']:.3f}")
        point_handles.append(
            Line2D([0], [0], linestyle="-", color=colors[pi % len(colors)], label=label)
        )

    if map_handles:
        legend_maps = ax.legend(handles=map_handles, title="Map (linestyle)",
                                loc="upper left", fontsize="small", framealpha=0.9)
        ax.add_artist(legend_maps)
    if point_handles:
        ax.legend(handles=point_handles, title="Point (color)",
                  loc="lower right", fontsize="small", framealpha=0.9)

    if size and output:
        try:
            w_px, h_px = [int(s) for s in size.split(",")]
            fig.set_size_inches(w_px / 100.0, h_px / 100.0)
        except Exception:
            pass

    if output:
        fig.savefig(output, bbox_inches="tight")
    else:
        plt.show()

def _parse_maps(opt):
    # GRASS may pass list (multiple=yes) or a comma-separated string
    if isinstance(opt, list):
        maps = []
        for m in opt:
            maps.extend([x.strip() for x in str(m).split(",") if x.strip()])
        return maps
    return [x.strip() for x in str(opt).split(",") if x.strip()]

def _parse_coordinates(opt):
    # Accept list or comma-separated string; return flat token list of strings
    if isinstance(opt, list):
        tokens = []
        for part in opt:
            tokens.extend([t for t in str(part).split(",") if t.strip() != ""])
    else:
        tokens = [t for t in str(opt).split(",") if t.strip() != ""]
    if len(tokens) % 2 != 0:
        gs.fatal("Coordinates list must contain an even number of values (E,N pairs)")
    return tokens

def main(options, flags):
    maps = _parse_maps(options["map"])
    tokens = _parse_coordinates(options["coordinates"])

    gs.use_temp_region()

    datasets = []
    for mapname in maps:
        gs.run_command("g.region", raster_3d=mapname)
        band_count = _band_count(mapname)
        wavelengths, fwhm = _band_wavelengths(mapname, band_count)

        points = []
        for i in range(0, len(tokens), 2):
            try:
                e = float(tokens[i])
                n = float(tokens[i + 1])
            except ValueError:
                gs.fatal(f"Non-numeric coordinate at position {i}: {tokens[i]}, {tokens[i+1]}")
            values = _sample_all_bands_at_point(mapname, e, n, band_count)
            points.append({"x": e, "y": n, "values": values})

        datasets.append({"map": mapname, "wavelength_nm": wavelengths, "points": points})

    # If -p (print) is given, emit JSON instead of plotting
    if flags.get("p"):
        results = {"maps": [ds["map"] for ds in datasets], "datasets": datasets}
        print(json.dumps(results, ensure_ascii=False))
        return

    # Default behavior: plot
    _plot_results_multi(
        datasets=datasets,
        title="Spectra",
        xlabel="Wavelength (nm)",
        ylabel="Value",
        output=options.get("output"),
        size=options.get("size"),
    )

if __name__ == "__main__":
    options, flags = gs.parser()
    main(options, flags)
