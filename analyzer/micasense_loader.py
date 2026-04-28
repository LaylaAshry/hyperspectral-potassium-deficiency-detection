import os
import numpy as np
from PIL import Image
from PIL.ExifTags import TAGS


# RedEdge-3 (4 bands) — matches sample.micasense.com
REDEDGE_3_WAVELENGTHS = {
    1: 480,   # Blue    _1.tif
    2: 550,   # Green   _2.tif
    3: 670,   # Red     _3.tif
    4: 850,   # NIR     _4.tif
}

# RedEdge-MX (5 bands)
REDEDGE_MX_WAVELENGTHS = {
    1: 475,   # Blue       _1.tif
    2: 560,   # Green      _2.tif
    3: 668,   # Red        _3.tif
    4: 717,   # Red Edge   _4.tif
    5: 842,   # NIR        _5.tif
}


def load_micasense_capture(folder_or_files):
    """
    Loads a MicaSense capture from either:
      - A folder containing the band TIFFs
      - A list of file paths

    Supports both RedEdge-3 (4 bands) and RedEdge-MX (5 bands).
    Automatically detects which camera based on file count.

    Returns:
        wavelengths  — np.array of wavelength values in nm
        reflectances — np.array of average reflectance values (0-1)
        gps          — dict with 'lat', 'lon', 'alt' or None
        capture_id   — string e.g. 'IMG_0001'
    """
    if isinstance(folder_or_files, (list, tuple)):
        tif_files = sorted(folder_or_files)
    else:
        tif_files = sorted([
            os.path.join(folder_or_files, f)
            for f in os.listdir(folder_or_files)
            if f.lower().endswith('.tif')
        ])

    if len(tif_files) == 0:
        raise ValueError("No TIFF files found.")
    if len(tif_files) not in (4, 5):
        raise ValueError(
            f"Expected 4 or 5 band TIFFs, found {len(tif_files)}. "
            f"Make sure you select all band files for one capture.")

    # Pick wavelength map based on band count
    band_map = (REDEDGE_3_WAVELENGTHS if len(tif_files) == 4
                else REDEDGE_MX_WAVELENGTHS)

    wavelengths  = []
    reflectances = []
    gps          = None
    capture_id   = None

    for i, tif_path in enumerate(tif_files):
        band_num   = _get_band_number(tif_path)
        wavelength = band_map.get(band_num, list(band_map.values())[i])
        wavelengths.append(wavelength)

        img = Image.open(tif_path)
        arr = np.array(img, dtype=np.float32)

        # Raw 16-bit DN → normalize to 0-1 reflectance
        reflectance = np.mean(arr) / 65535.0
        reflectances.append(reflectance)

        if gps is None:
            gps = _extract_gps(img)

        if capture_id is None:
            basename   = os.path.basename(tif_path)
            parts      = os.path.splitext(basename)[0].rsplit('_', 1)
            capture_id = parts[0] if len(parts) == 2 else basename

    order        = np.argsort(wavelengths)
    wavelengths  = np.array(wavelengths)[order]
    reflectances = np.array(reflectances)[order]

    return wavelengths, reflectances, gps, capture_id


def _get_band_number(tif_path):
    """Extract band number from filename e.g. IMG_0001_3.tif → 3"""
    basename = os.path.basename(tif_path)
    name     = os.path.splitext(basename)[0]
    try:
        return int(name.split('_')[-1])
    except ValueError:
        return 1


def _extract_gps(img):
    """Extract GPS coordinates from TIFF EXIF metadata."""
    try:
        exif_data = img._getexif()
        if exif_data is None:
            return None

        exif = {TAGS.get(k, k): v for k, v in exif_data.items()}
        gps_info = exif.get('GPSInfo')
        if not gps_info:
            return None

        from PIL.ExifTags import GPSTAGS
        gps = {GPSTAGS.get(k, k): v for k, v in gps_info.items()}

        lat = _convert_gps(gps.get('GPSLatitude'),
                           gps.get('GPSLatitudeRef'))
        lon = _convert_gps(gps.get('GPSLongitude'),
                           gps.get('GPSLongitudeRef'))
        alt = _convert_alt(gps.get('GPSAltitude'))

        if lat is None or lon is None:
            return None

        return {'lat': lat, 'lon': lon, 'alt': alt}

    except Exception:
        return None


def _convert_gps(coord, ref):
    """Convert GPS DMS tuple to decimal degrees."""
    if coord is None or ref is None:
        return None
    try:
        degrees = float(coord[0])
        minutes = float(coord[1])
        seconds = float(coord[2])
        decimal = degrees + minutes / 60 + seconds / 3600
        if ref in ['S', 'W']:
            decimal = -decimal
        return round(decimal, 7)
    except Exception:
        return None


def _convert_alt(alt):
    """Convert GPS altitude to float metres."""
    if alt is None:
        return None
    try:
        return round(float(alt), 1)
    except Exception:
        return None


def downsample_baseline_to_rededge(wl_full, r_full,
                                    n_bands=4):
    """
    Downsamples a full spectrometer baseline to either
    4 bands (RedEdge-3) or 5 bands (RedEdge-MX) by interpolating.

    Use this to convert your existing healthy.txt / deficient.txt
    baselines until you can re-capture them with the RedEdge.

    Args:
        wl_full  — full wavelength array from spectrometer
        r_full   — full reflectance array from spectrometer
        n_bands  — 4 for RedEdge-3, 5 for RedEdge-MX

    Returns:
        target_wavelengths — np.array of band center wavelengths
        r_downsampled      — np.array of interpolated reflectances (0-1)
    """
    from scipy.interpolate import interp1d

    band_map = (REDEDGE_3_WAVELENGTHS if n_bands == 4
                else REDEDGE_MX_WAVELENGTHS)
    target_wavelengths = np.array(sorted(band_map.values()))

    f = interp1d(wl_full, r_full, kind='linear',
                 bounds_error=False, fill_value='extrapolate')
    r_downsampled = f(target_wavelengths)

    # Normalize to 0-1 to match RedEdge DN→reflectance scale
    r_min, r_max = r_downsampled.min(), r_downsampled.max()
    if r_max > r_min:
        r_downsampled = (r_downsampled - r_min) / (r_max - r_min)

    return target_wavelengths, r_downsampled