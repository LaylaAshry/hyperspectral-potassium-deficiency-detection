import os
import time
import glob

import pandas as pd

from analyzer.loader import load_spectrum
from analyzer.calibration import calibrate
from analyzer.classifier import classify
from analyzer.plotting import print_result, plot_comparison


def process_file(file_path, wl_h, r_h, wl_d, r_d, wl_white, r_white,
                 output_log, plot_results=False):
    """
    Full pipeline for one capture file:
      1. Load raw spectrum
      2. Calibrate against white reference
      3. Classify against healthy/deficient baselines
      4. Log result to CSV
      5. Optionally plot
    """
    print(f"\n  Processing: {os.path.basename(file_path)}")

    try:
        wl_raw, r_raw = load_spectrum(file_path)
    except Exception as e:
        print(f"  ✗ Could not read file: {e}")
        return None

    wl_cal, r_cal = calibrate(wl_raw, r_raw, wl_white, r_white)
    result = classify(wl_cal, r_cal, wl_h, r_h, wl_d, r_d)
    print_result(os.path.basename(file_path), result)

    timestamp = time.strftime('%Y-%m-%d %H:%M:%S',
                              time.localtime(os.path.getmtime(file_path)))
    row = {
        'timestamp':                timestamp,
        'file':                     os.path.basename(file_path),
        'classification':           result['classification'],
        'healthy_similarity_pct':   result['healthy_similarity_pct'],
        'deficient_similarity_pct': result['deficient_similarity_pct'],
        'margin_pct':               result['margin_pct'],
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

    df = pd.DataFrame([row])
    df.to_csv(output_log, mode='a',
              header=not os.path.exists(output_log),
              index=False)
    print(f"  ✓ Logged to {output_log}")

    if plot_results:
        plot_comparison(wl_cal, r_cal, wl_h, r_h, wl_d, r_d,
                        os.path.basename(file_path), result)
    return result


def run_pipeline(input_folder, output_log, healthy_base, deficient_base,
                 white_ref_file, plot_results=False):
    """
    Processes all .txt and .csv files in input_folder.
    Loads baselines once, then classifies each file and logs results.
    """
    print("Loading baselines and white reference...")
    try:
        wl_h, r_h = load_spectrum(healthy_base)
        wl_d, r_d = load_spectrum(deficient_base)
        wl_w, r_w = load_spectrum(white_ref_file)
        print(f"  ✓ Healthy   — {len(wl_h)} bands")
        print(f"  ✓ Deficient — {len(wl_d)} bands")
        print(f"  ✓ White ref — {len(wl_w)} bands")
    except Exception as e:
        print(f"  ✗ Error loading baselines: {e}")
        return

    files = sorted(
        glob.glob(os.path.join(input_folder, "*.txt")) +
        glob.glob(os.path.join(input_folder, "*.csv"))
    )

    if not files:
        print(f"  No files found in {input_folder}")
        return

    print(f"\n  Found {len(files)} file(s) to process...\n")

    all_results = {}
    for file_path in files:
        result = process_file(
            file_path, wl_h, r_h, wl_d, r_d, wl_w, r_w,
            output_log, plot_results
        )
        if result:
            all_results[os.path.basename(file_path)] = result

    counts = {'Healthy': 0, 'Deficient': 0, 'Uncertain': 0}
    for r in all_results.values():
        counts[r['classification']] += 1

    print(f"\n{'━' * 50}")
    print(f"  SUMMARY — {len(all_results)} files processed")
    print(f"  ✅ Healthy   : {counts['Healthy']}")
    print(f"  ❌ Deficient : {counts['Deficient']}")
    print(f"  ⚠️  Uncertain : {counts['Uncertain']}")
    print(f"{'━' * 50}\n")