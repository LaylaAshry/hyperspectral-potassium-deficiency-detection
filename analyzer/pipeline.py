import os
import time
import glob

import numpy as np
import pandas as pd

from analyzer.loader import load_spectrum
from analyzer.micasense_loader import (load_micasense_capture,
                                        downsample_baseline_to_rededge)
from analyzer.classifier import classify
from analyzer.plotting import print_result, plot_comparison
from analyzer.map_display import generate_map


# ── Baseline loading ──────────────────────────────────────────────

def load_baselines_for_rededge(healthy_path, deficient_path,
                                n_bands=4):
    """
    Loads full spectrometer baselines and downsamples them to
    match the RedEdge band count (4 for RedEdge-3, 5 for MX).
    """
    wl_h_full, r_h_full = load_spectrum(healthy_path)
    wl_d_full, r_d_full = load_spectrum(deficient_path)
    wl_h, r_h = downsample_baseline_to_rededge(
        wl_h_full, r_h_full, n_bands)
    wl_d, r_d = downsample_baseline_to_rededge(
        wl_d_full, r_d_full, n_bands)
    return wl_h, r_h, wl_d, r_d


# ── Capture grouping ──────────────────────────────────────────────

def group_into_captures(tif_files):
    """
    Groups TIFF files into captures by matching the capture number
    in the filename.
    e.g. IMG_0001_1.tif through IMG_0001_4.tif becomes one capture.
    Accepts both 4-band (RedEdge-3) and 5-band (RedEdge-MX) captures.
    """
    captures = {}
    for path in tif_files:
        basename   = os.path.basename(path)
        name       = os.path.splitext(basename)[0]
        parts      = name.rsplit('_', 1)
        capture_id = parts[0] if len(parts) == 2 else name
        if capture_id not in captures:
            captures[capture_id] = []
        captures[capture_id].append(path)

    complete   = {}
    incomplete = {}
    for k, v in captures.items():
        if len(v) in (4, 5):
            complete[k] = sorted(v)
        else:
            incomplete[k] = v

    if incomplete:
        print(f"  ⚠️  Skipping {len(incomplete)} incomplete "
              f"capture(s): {list(incomplete.keys())}")

    return complete


def _detect_band_count(tif_files):
    """
    Detects whether files are from a 4-band or 5-band camera
    by checking the highest band suffix found in filenames.
    """
    max_band = 4
    for path in tif_files:
        name = os.path.splitext(os.path.basename(path))[0]
        try:
            band = int(name.split('_')[-1])
            max_band = max(max_band, band)
        except ValueError:
            pass
    return 5 if max_band >= 5 else 4


# ── Single capture processing ─────────────────────────────────────

def process_micasense_capture(capture_files, wl_h, r_h, wl_d, r_d,
                               output_log, plot_results=False):
    """
    Full pipeline for one RedEdge capture (4 or 5 TIFF files):
      1. Load and average reflectance per band
      2. Extract GPS coordinates from EXIF
      3. Classify against downsampled baselines
      4. Log result to CSV
      5. Optionally plot
    """
    try:
        wl_s, r_s, gps, capture_id = load_micasense_capture(
            capture_files)
    except Exception as e:
        print(f"  ✗ Could not load capture: {e}")
        return None

    print(f"\n  Processing: {capture_id}")

    result = classify(wl_s, r_s, wl_h, r_h, wl_d, r_d)
    print_result(capture_id, result)

    if gps:
        alt_str = (f"  Alt: {gps['alt']}m"
                   if gps.get('alt') else '')
        print(f"  📍 GPS: {gps['lat']}, {gps['lon']}{alt_str}")
    else:
        print("  📍 GPS: not found in EXIF")

    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    lat = gps['lat'] if gps else ''
    lon = gps['lon'] if gps else ''
    alt = gps['alt'] if gps else ''

    row = {
        'timestamp':                timestamp,
        'capture_id':               capture_id,
        'classification':           result['classification'],
        'healthy_similarity_pct':   result['healthy_similarity_pct'],
        'deficient_similarity_pct': result['deficient_similarity_pct'],
        'margin_pct':               result['margin_pct'],
        'lat':                      lat,
        'lon':                      lon,
        'alt_m':                    alt,
        'sam_vs_healthy_deg':       result['full_spectrum']['sam_vs_healthy_deg'],
        'sam_vs_deficient_deg':     result['full_spectrum']['sam_vs_deficient_deg'],
        'pearson_vs_healthy':       result['full_spectrum']['pearson_vs_healthy'],
        'pearson_vs_deficient':     result['full_spectrum']['pearson_vs_deficient'],
        'band_visible_H':           result['band_details']['Visible']['similarity_to_healthy'],
        'band_visible_D':           result['band_details']['Visible']['similarity_to_deficient'],
        'band_rededge_H':           result['band_details']['Red Edge']['similarity_to_healthy'],
        'band_rededge_D':           result['band_details']['Red Edge']['similarity_to_deficient'],
        'band_nir_H':               result['band_details']['NIR']['similarity_to_healthy'],
        'band_nir_D':               result['band_details']['NIR']['similarity_to_deficient'],
    }

    os.makedirs(os.path.dirname(output_log), exist_ok=True)
    df = pd.DataFrame([row])
    df.to_csv(output_log, mode='a',
              header=not os.path.exists(output_log),
              index=False)
    print(f"  ✓ Logged to {output_log}")

    if plot_results:
        plot_comparison(wl_s, r_s, wl_h, r_h, wl_d, r_d,
                        capture_id, result)

    return {
        'capture_id':     capture_id,
        'classification': result['classification'],
        'healthy_pct':    result['healthy_similarity_pct'],
        'deficient_pct':  result['deficient_similarity_pct'],
        'margin_pct':     result['margin_pct'],
        'lat':            gps['lat'] if gps else None,
        'lon':            gps['lon'] if gps else None,
        'alt':            gps['alt'] if gps else None,
    }


# ── Batch pipeline ────────────────────────────────────────────────

def run_pipeline(input_folder, output_log, healthy_base,
                 deficient_base, plot_results=False):
    """
    Processes all RedEdge captures in input_folder.
    Groups TIFF files into captures by filename.
    Generates a field map if GPS data is available.
    Supports both RedEdge-3 (4 bands) and RedEdge-MX (5 bands).
    """
    all_tifs = sorted(
        glob.glob(os.path.join(input_folder, "*.tif")) +
        glob.glob(os.path.join(input_folder, "*.TIF"))
    )

    if not all_tifs:
        print(f"  No TIFF files found in {input_folder}")
        return []

    # Detect band count from files
    n_bands = _detect_band_count(all_tifs)
    print(f"  Detected {n_bands}-band camera "
          f"({'RedEdge-MX' if n_bands == 5 else 'RedEdge-3'})")

    print("Loading and downsampling baselines...")
    try:
        wl_h, r_h, wl_d, r_d = load_baselines_for_rededge(
            healthy_base, deficient_base, n_bands)
        wl_list = [int(w) for w in wl_h]
        print(f"  ✓ Baselines downsampled to "
              f"{n_bands} bands: {wl_list} nm")
    except Exception as e:
        print(f"  ✗ Error loading baselines: {e}")
        return []

    captures = group_into_captures(all_tifs)
    print(f"\n  Found {len(captures)} complete "
          f"capture(s) to process...\n")

    all_results = []
    for capture_id, files in captures.items():
        result = process_micasense_capture(
            files, wl_h, r_h, wl_d, r_d,
            output_log, plot_results)
        if result:
            all_results.append(result)

    gps_results = [r for r in all_results if r.get('lat')]
    if gps_results:
        map_path = os.path.join(
            os.path.dirname(output_log), 'field_map.html')
        generate_map(all_results, map_path)
        print(f"  🗺  Field map saved → {map_path}")

    counts = {'Healthy': 0, 'Deficient': 0, 'Uncertain': 0}
    for r in all_results:
        counts[r['classification']] += 1

    print(f"\n{'━' * 50}")
    print(f"  SUMMARY — {len(all_results)} captures processed")
    print(f"  ✅ Healthy   : {counts['Healthy']}")
    print(f"  ❌ Deficient : {counts['Deficient']}")
    print(f"  ⚠️  Uncertain : {counts['Uncertain']}")
    print(f"{'━' * 50}\n")

    return all_results