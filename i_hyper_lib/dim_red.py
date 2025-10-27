#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from sklearn.decomposition import PCA, KernelPCA
from sklearn.kernel_approximation import Nystroem
import joblib
import re


def _parse_band_intervals(band_string, wavelengths):
    """Select subset of bands by intervals or single nm values."""
    if not band_string:
        return np.arange(len(wavelengths))
    selection = np.zeros(len(wavelengths), dtype=bool)
    for part in band_string.split(","):
        part = part.strip()
        if "-" in part:
            start, end = map(float, part.split("-"))
            selection |= (wavelengths >= start) & (wavelengths <= end)
        else:
            val = float(part)
            selection |= np.isclose(wavelengths, val, atol=5)
    return np.where(selection)[0]


def _apply_dimensionality_reduction(flat_data, method=None, n_components=0,
                                    kernel="rbf", gamma=0.01, degree=3,
                                    bands=None, wavelengths=None,
                                    export_path=None,
                                    chunk_size=None, memory_limit_gb=8):
    """Apply PCA, KPCA, or Nystroem+PCA to flattened hyperspectral data, NaN-safe and pixel-wise."""
    info = {}
    X = flat_data

    # Optional band filtering
    if bands and wavelengths is not None:
        idx = _parse_band_intervals(bands, np.array(wavelengths))
        if idx.size == 0:
            raise ValueError("No wavelengths matched given intervals.")
        X = X[:, idx]

    if not method:
        return X, info

    # Validate components
    if n_components <= 0 or n_components > X.shape[1]:
        n_components = min(10, X.shape[1])

    # Identify valid spectra (rows without NaNs)
    valid_mask = ~np.isnan(X).any(axis=1)
    if valid_mask.sum() == 0:
        raise ValueError("No valid spectra for dimensionality reduction.")
    X_valid = X[valid_mask]

    # Determine chunk size if not specified
    if chunk_size is None:
        bytes_per_val = 4
        total_cols = X_valid.shape[1]
        chunk_size = int((memory_limit_gb * (1024 ** 3)) / (bytes_per_val * total_cols))
        chunk_size = max(5000, min(chunk_size, X_valid.shape[0]))
    chunks = range(0, X_valid.shape[0], chunk_size)

    X_red_parts = []
    info = {}

    # Process each chunk separately
    for start in chunks:
        end = min(start + chunk_size, X_valid.shape[0])
        X_chunk = X_valid[start:end]

        if method == "PCA":
            model = PCA(n_components=n_components)
            X_red_chunk = model.fit_transform(X_chunk)
            info["explained_variance_ratio"] = model.explained_variance_ratio_

        elif method == "KPCA":
            model = KernelPCA(n_components=n_components, kernel=kernel,
                              gamma=gamma, degree=degree, fit_inverse_transform=False)
            X_red_chunk = model.fit_transform(X_chunk)
            info.update({
                "kernel": kernel,
                "gamma": gamma,
                "degree": degree,
                "n_components": n_components
            })

        elif method == "Nystroem":
            feature_map = Nystroem(kernel=kernel, gamma=gamma,
                                   degree=degree, n_components=n_components)
            X_mapped = feature_map.fit_transform(X_chunk)
            pca = PCA(n_components=min(n_components, X_mapped.shape[1]))
            X_red_chunk = pca.fit_transform(X_mapped)
            info.update({
                "kernel": kernel,
                "gamma": gamma,
                "degree": degree,
                "n_components": n_components,
                "explained_variance_ratio": pca.explained_variance_ratio_
            })
        else:
            raise ValueError(f"Unsupported dimensionality reduction method: {method}")

        X_red_parts.append(X_red_chunk.astype(np.float32))

    # Combine all chunk results
    X_valid_red = np.vstack(X_red_parts)

    # Reinsert transformed spectra into full array with NaNs for invalid pixels
    X_red = np.full((X.shape[0], X_valid_red.shape[1]), np.nan, dtype=np.float32)
    X_red[valid_mask] = X_valid_red

    # Export model(s)
    if export_path:
        joblib.dump(model if method != "Nystroem" else (feature_map, pca), export_path)

    return X_red.astype(np.float32), info
