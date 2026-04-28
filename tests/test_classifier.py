import numpy as np
import pytest

from analyzer.loader import load_spectrum
from analyzer.classifier import classify
from analyzer.micasense_loader import downsample_baseline_to_rededge


# ── Helpers ───────────────────────────────────────────────────────

def make_sample(wl_h, r_h, wl_d, r_d, hw, dw, seed=42):
    """Generate a synthetic spectrum blended from healthy/deficient."""
    from scipy.interpolate import interp1d
    np.random.seed(seed)
    f_d = interp1d(wl_d, r_d, bounds_error=False, fill_value='extrapolate')
    r_d_resampled = f_d(wl_h)
    noise = np.random.normal(0, 0.01, len(wl_h))
    r = np.clip(hw * r_h + dw * r_d_resampled + noise, 0, None)
    return wl_h, r


# ── Fixtures ──────────────────────────────────────────────────────

@pytest.fixture(scope='module')
def baselines():
    wl_h_full, r_h_full = load_spectrum('./baselines/healthy.txt')
    wl_d_full, r_d_full = load_spectrum('./baselines/deficient.txt')
    wl_h, r_h = downsample_baseline_to_rededge(wl_h_full, r_h_full)
    wl_d, r_d = downsample_baseline_to_rededge(wl_d_full, r_d_full)
    return wl_h, r_h, wl_d, r_d


# ── Tests ─────────────────────────────────────────────────────────

def test_healthy_classified_correctly(baselines):
    wl_h, r_h, wl_d, r_d = baselines
    wl_s, r_s = make_sample(wl_h, r_h, wl_d, r_d, hw=0.95, dw=0.05)
    result = classify(wl_s, r_s, wl_h, r_h, wl_d, r_d)
    assert result['classification'] == 'Healthy'
    assert result['healthy_similarity_pct'] > result['deficient_similarity_pct']


def test_deficient_classified_correctly(baselines):
    wl_h, r_h, wl_d, r_d = baselines
    wl_s, r_s = make_sample(wl_h, r_h, wl_d, r_d, hw=0.05, dw=0.95)
    result = classify(wl_s, r_s, wl_h, r_h, wl_d, r_d)
    assert result['classification'] == 'Deficient'
    assert result['deficient_similarity_pct'] > result['healthy_similarity_pct']


def test_result_has_required_keys(baselines):
    wl_h, r_h, wl_d, r_d = baselines
    wl_s, r_s = make_sample(wl_h, r_h, wl_d, r_d, hw=0.5, dw=0.5)
    result = classify(wl_s, r_s, wl_h, r_h, wl_d, r_d)
    assert 'classification' in result
    assert 'healthy_similarity_pct' in result
    assert 'deficient_similarity_pct' in result
    assert 'margin_pct' in result
    assert 'band_details' in result
    assert 'full_spectrum' in result


def test_similarities_sum_to_100(baselines):
    wl_h, r_h, wl_d, r_d = baselines
    wl_s, r_s = make_sample(wl_h, r_h, wl_d, r_d, hw=0.5, dw=0.5)
    result = classify(wl_s, r_s, wl_h, r_h, wl_d, r_d)
    total = result['healthy_similarity_pct'] + result['deficient_similarity_pct']
    assert abs(total - 100.0) < 0.1


def test_downsample_to_rededge_4_bands():
    wl_full, r_full = load_spectrum('./baselines/healthy.txt')
    wl_re, r_re = downsample_baseline_to_rededge(wl_full, r_full, n_bands=4)
    assert len(wl_re) == 4
    assert len(r_re) == 4
    assert all(0 <= v <= 1 for v in r_re)


def test_downsample_to_rededge_5_bands():
    wl_full, r_full = load_spectrum('./baselines/healthy.txt')
    wl_re, r_re = downsample_baseline_to_rededge(wl_full, r_full, n_bands=5)
    assert len(wl_re) == 5
    assert len(r_re) == 5
    assert all(0 <= v <= 1 for v in r_re)