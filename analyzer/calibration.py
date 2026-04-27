import numpy as np
from scipy.interpolate import interp1d


def calibrate(wl_raw, r_raw, wl_white, r_white):
    """
    Converts raw camera counts to reflectance by dividing by the
    white reference. Replicates the Spectronon calibration step.

    Steps:
      1. Resample the raw spectrum onto the white reference wavelength grid
      2. Divide raw / white  →  values near 1.0 = highly reflective

    Args:
        wl_raw:    wavelengths of the raw capture
        r_raw:     raw reflectance counts from camera
        wl_white:  wavelengths of the white reference
        r_white:   reflectance of the white reference (spectralon panel)

    Returns:
        wl_cal, r_cal — calibrated wavelengths and reflectance
    """
    f = interp1d(wl_raw, r_raw, kind='linear',
                 bounds_error=False, fill_value='extrapolate')
    r_raw_resampled = f(wl_white)

    r_calibrated = np.divide(
        r_raw_resampled, r_white,
        out=np.zeros_like(r_raw_resampled),
        where=r_white != 0
    )
    return wl_white, r_calibrated