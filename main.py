import argparse
import os

from analyzer.pipeline import run_pipeline

# ══════════════════════════════════════════════════════════════════
# DEFAULT PATHS — edit these or override with command line arguments
# ══════════════════════════════════════════════════════════════════
DEFAULTS = {
    'input':      './camera_data',
    'output':     './results/field_results.csv',
    'healthy':    './baselines/healthy.txt',
    'deficient':  './baselines/deficient.txt',
    'white_ref':  './baselines/white_reference.txt',
}


def main():
    parser = argparse.ArgumentParser(
        description='Mango leaf potassium deficiency detector.'
    )
    parser.add_argument('--input',     default=DEFAULTS['input'],
                        help='Folder containing drone capture files')
    parser.add_argument('--output',    default=DEFAULTS['output'],
                        help='Path to save CSV results log')
    parser.add_argument('--healthy',   default=DEFAULTS['healthy'],
                        help='Healthy baseline spectrum file')
    parser.add_argument('--deficient', default=DEFAULTS['deficient'],
                        help='Deficient baseline spectrum file')
    parser.add_argument('--white_ref', default=DEFAULTS['white_ref'],
                        help='White reference calibration file')
    parser.add_argument('--plot',      action='store_true',
                        help='Show plot for each file processed')
    args = parser.parse_args()

    # Create output folder if it doesn't exist
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    os.makedirs(args.input, exist_ok=True)

    run_pipeline(
        input_folder   = args.input,
        output_log     = args.output,
        healthy_base   = args.healthy,
        deficient_base = args.deficient,
        white_ref_file = args.white_ref,
        plot_results   = args.plot,
    )


if __name__ == '__main__':
    main()