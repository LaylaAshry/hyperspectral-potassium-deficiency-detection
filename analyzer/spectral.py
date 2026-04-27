import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import pearsonr


def normalize(r):
    """Scales reflectance values to 0-1 so files with different
    brightness levels can be fairly compared."""
    lo, hi = r.min(), r.max()
    return (r - lo) / (hi - lo) if hi != lo else np.zeros_like(r)


def resample(wl_src, r_src, wl_target):
    """Interpolates a spectrum onto a new set of wavelength points."""
    f = interp1d(wl_src, r_src, kind='linear',
                 bounds_error=False, fill_value='extrapolate')
    return f(wl_target)


def common_grid(wl1, r1, wl2, r2, n=500):
    """Puts two spectra on the same wavelength grid for comparison."""
    lo = max(wl1.min(), wl2.min())
    hi = min(wl1.max(), wl2.max())
    wl = np.linspace(lo, hi, n)
    f1 = interp1d(wl1, r1, kind='linear', bounds_error=False, fill_value='extrapolate')
    f2 = interp1d(wl2, r2, kind='linear', bounds_error=False, fill_value='extrapolate')
    return wl, f1(wl), f2(wl)


def sam_similarity(a, b):
    """Spectral Angle Mapper — compares the shape of two spectra.
    Score of 1.0 = identical, 0.0 = completely different."""
    dot  = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm < 1e-12:
        return 0.0
    angle = float(np.degrees(np.arccos(np.clip(dot / norm, -1, 1))))
    return 1.0 - angle / 90.0


def corr_similarity(a, b):
    """Pearson correlation rescaled to 0-1."""
    if np.std(a) < 1e-10 or np.std(b) < 1e-10:
        return 0.5
    r, _ = pearsonr(a, b)
    return (float(r) + 1.0) / 2.0


def window_similarity(wl, r_s, r_b, lo, hi):
    """Combined SAM + Pearson similarity within one wavelength band."""
    mask = (wl >= lo) & (wl <= hi)
    if mask.sum() < 4:
        return 0.5
    a, b = r_s[mask], r_b[mask]
    return 0.6 * sam_similarity(a, b) + 0.4 * corr_similarity(a, b)