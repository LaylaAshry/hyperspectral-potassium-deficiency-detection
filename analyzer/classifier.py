import numpy as np
from scipy.stats import pearsonr

from analyzer.spectral import normalize, resample, window_similarity

# ── Spectral windows calibrated for mango potassium deficiency ───
# Visible is most discriminative (107% mean diff between baselines),
# followed by red-edge (76%), then NIR (6%).
SPECTRAL_WINDOWS = [
    {'name': 'Visible',  'lo': 400, 'hi': 700,  'weight': 0.55},
    {'name': 'Red Edge', 'lo': 680, 'hi': 750,  'weight': 0.30},
    {'name': 'NIR',      'lo': 750, 'hi': 1010, 'weight': 0.15},
]

# Minimum margin (percentage points) for a confident classification
CONFIDENCE_THRESHOLD = 2.0


def classify(wl_s, r_s, wl_h, r_h, wl_d, r_d):
    """
    Compares a sample to both baselines across three spectral bands
    and returns Healthy, Deficient, or Uncertain.
    """
    lo = max(wl_s.min(), wl_h.min(), wl_d.min())
    hi = min(wl_s.max(), wl_h.max(), wl_d.max())
    wl = np.linspace(lo, hi, 600)

    rs = normalize(resample(wl_s, r_s, wl))
    rh = normalize(resample(wl_h, r_h, wl))
    rd = normalize(resample(wl_d, r_d, wl))

    score_h = score_d = 0.0
    band_details = {}

    for w in SPECTRAL_WINDOWS:
        sh = window_similarity(wl, rs, rh, w['lo'], w['hi'])
        sd = window_similarity(wl, rs, rd, w['lo'], w['hi'])
        score_h += w['weight'] * sh
        score_d += w['weight'] * sd
        band_details[w['name']] = {
            'similarity_to_healthy':   round(sh * 100, 1),
            'similarity_to_deficient': round(sd * 100, 1),
        }

    total = score_h + score_d
    pct_h = (score_h / total * 100) if total > 0 else 50.0
    pct_d = (score_d / total * 100) if total > 0 else 50.0
    margin = abs(pct_h - pct_d)

    if margin < CONFIDENCE_THRESHOLD:
        label = 'Uncertain'
    elif pct_h > pct_d:
        label = 'Healthy'
    else:
        label = 'Deficient'

    def sam_deg(a, b):
        n = np.linalg.norm(a) * np.linalg.norm(b)
        if n < 1e-12:
            return 90.0
        return float(np.degrees(np.arccos(np.clip(np.dot(a, b) / n, -1, 1))))

    r_hr, _ = pearsonr(rs, rh)
    r_dr, _ = pearsonr(rs, rd)

    return {
        'classification':            label,
        'healthy_similarity_pct':    round(pct_h, 2),
        'deficient_similarity_pct':  round(pct_d, 2),
        'margin_pct':                round(margin, 2),
        'band_details':              band_details,
        'full_spectrum': {
            'sam_vs_healthy_deg':    round(sam_deg(rs, rh), 2),
            'sam_vs_deficient_deg':  round(sam_deg(rs, rd), 2),
            'pearson_vs_healthy':    round(float(r_hr), 4),
            'pearson_vs_deficient':  round(float(r_dr), 4),
        },
    }