import re
import numpy as np


def parse_spectranon(text):
    """Reads SpectraNon format: Wavelength = [...] / Reflectance = [...]"""
    wl_m  = re.search(r'Wavelength\s*=\s*\[([^\]]+)\]',  text)
    ref_m = re.search(r'Reflectance\s*=\s*\[([^\]]+)\]', text)
    if not wl_m or not ref_m:
        raise ValueError("Could not find Wavelength/Reflectance arrays.")
    wl  = np.array([float(x) for x in wl_m.group(1).split(',')])
    ref = np.array([float(x) for x in ref_m.group(1).split(',')])
    return wl, ref


def parse_tabular(text):
    """Reads tab- or comma-separated two-column files."""
    wl, ref = [], []
    for line in text.strip().splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        parts = re.split(r'[\t,]+', line)
        if len(parts) >= 2:
            try:
                wl.append(float(parts[0]))
                ref.append(float(parts[1]))
            except ValueError:
                continue
    if not wl:
        raise ValueError("No numeric data found.")
    return np.array(wl), np.array(ref)


def load_spectrum(path):
    """Auto-detects file format and loads it."""
    with open(path, 'r') as f:
        text = f.read()
    if 'Wavelength' in text and '=' in text:
        return parse_spectranon(text)
    return parse_tabular(text)