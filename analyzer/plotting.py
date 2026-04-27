import matplotlib.pyplot as plt

from analyzer.spectral import common_grid


def print_result(filename, result):
    """Prints a formatted summary of a classification result."""
    label = result['classification']
    icons = {'Healthy': '✅', 'Deficient': '❌', 'Uncertain': '⚠️'}
    print(f"\n{'═' * 55}")
    print(f"  File: {filename}")
    print(f"{'═' * 55}")
    print(f"  Result : {icons.get(label, '?')} {label}")
    print(f"  Margin : {result['margin_pct']:.1f} pp\n")
    ph = result['healthy_similarity_pct']
    pd = result['deficient_similarity_pct']
    bar = lambda p: '█' * int(p / 100 * 28) + '░' * (28 - int(p / 100 * 28))
    print(f"  Healthy    {bar(ph)}  {ph:.1f}%")
    print(f"  Deficient  {bar(pd)}  {pd:.1f}%\n")
    print("  Band breakdown:")
    for band, bd in result['band_details'].items():
        diff = bd['similarity_to_healthy'] - bd['similarity_to_deficient']
        arrow = '↑ Healthy' if diff > 0 else '↓ Deficient'
        print(f"    {band:12s}  H:{bd['similarity_to_healthy']:5.1f}%  "
              f"D:{bd['similarity_to_deficient']:5.1f}%  "
              f"Δ={abs(diff):.1f}pp  {arrow}")
    fs = result['full_spectrum']
    print(f"\n  SAM vs Healthy   : {fs['sam_vs_healthy_deg']:.2f}°  "
          f"(r={fs['pearson_vs_healthy']:.4f})")
    print(f"  SAM vs Deficient : {fs['sam_vs_deficient_deg']:.2f}°  "
          f"(r={fs['pearson_vs_deficient']:.4f})\n")


def plot_comparison(wl_s, r_s, wl_h, r_h, wl_d, r_d, title, result):
    """Plots sample spectrum against both baselines with a difference panel."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8), sharex=True,
                                   gridspec_kw={'height_ratios': [3, 1]})
    fig.patch.set_facecolor('#0f1117')
    for ax in (ax1, ax2):
        ax.set_facecolor('#161b22')
        ax.tick_params(colors='#8b949e')
        ax.spines[:].set_color('#30363d')

    ax1.plot(wl_h, r_h, color='#3fb950', lw=1.5,
             label='Healthy baseline', alpha=0.85)
    ax1.plot(wl_d, r_d, color='#f85149', lw=1.5,
             label='Deficient baseline', alpha=0.85)
    ax1.plot(wl_s, r_s, color='#58a6ff', lw=2.2,
             label=f'Sample → {result["classification"]}', zorder=5)

    for lo, hi, alpha in [(400, 700, 0.06), (680, 750, 0.09), (750, 1010, 0.04)]:
        ax1.axvspan(lo, hi, alpha=alpha, color='white')

    ax1.set_ylabel('Reflectance', color='#8b949e')
    ax1.legend(facecolor='#21262d', edgecolor='#30363d',
               labelcolor='white', fontsize=9)
    ax1.set_title(title, color='white', fontsize=12,
                  fontweight='bold', pad=8)

    wl_c, r_s_c, r_h_c = common_grid(wl_s, r_s, wl_h, r_h)
    diff = r_s_c - r_h_c
    ax2.fill_between(wl_c, diff, 0, where=(diff >= 0),
                     color='#f85149', alpha=0.6, label='Above healthy')
    ax2.fill_between(wl_c, diff, 0, where=(diff < 0),
                     color='#3fb950', alpha=0.6, label='Below healthy')
    ax2.axhline(0, color='#8b949e', lw=0.8)
    ax2.set_xlabel('Wavelength (nm)', color='#8b949e')
    ax2.set_ylabel('Δ Reflectance', color='#8b949e', fontsize=8)
    ax2.legend(facecolor='#21262d', edgecolor='#30363d',
               labelcolor='white', fontsize=8)

    plt.tight_layout()
    plt.show()