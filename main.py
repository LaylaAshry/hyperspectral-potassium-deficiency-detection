import argparse
import os

from analyzer.pipeline import run_pipeline


DEFAULTS = {
    'input':      './camera_data',
    'output':     './results/field_results.csv',
    'healthy':    './baselines/healthy.txt',
    'deficient':  './baselines/deficient.txt',
}


def main():
    parser = argparse.ArgumentParser(
        description='FAU Farm Owls SpectraSense — MicaSense RedEdge-MX pipeline.'
    )
    parser.add_argument('--input',     default=DEFAULTS['input'],
                        help='Folder containing RedEdge-MX TIFF captures')
    parser.add_argument('--output',    default=DEFAULTS['output'],
                        help='Path to save CSV results log')
    parser.add_argument('--healthy',   default=DEFAULTS['healthy'],
                        help='Healthy baseline spectrum file')
    parser.add_argument('--deficient', default=DEFAULTS['deficient'],
                        help='Deficient baseline spectrum file')
    parser.add_argument('--plot',      action='store_true',
                        help='Show spectral plot for each capture')
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    os.makedirs(args.input, exist_ok=True)

    run_pipeline(
        input_folder   = args.input,
        output_log     = args.output,
        healthy_base   = args.healthy,
        deficient_base = args.deficient,
        plot_results   = args.plot,
    )


if __name__ == '__main__':
    main()