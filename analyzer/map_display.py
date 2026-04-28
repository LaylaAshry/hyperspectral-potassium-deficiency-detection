import os
import folium


def generate_map(results, output_path='./results/field_map.html'):
    """
    Generates an interactive HTML map showing classification results
    as colour-coded markers that farmers can open in any browser.

    Args:
        results — list of dicts, each with:
                    'lat', 'lon', 'capture_id', 'classification',
                    'healthy_pct', 'deficient_pct', 'margin_pct'
        output_path — where to save the HTML file
    """
    if not results:
        return None

    # Filter to results that have GPS
    gps_results = [r for r in results if r.get('lat') and r.get('lon')]
    if not gps_results:
        return None

    # Centre map on average GPS position
    avg_lat = sum(r['lat'] for r in gps_results) / len(gps_results)
    avg_lon = sum(r['lon'] for r in gps_results) / len(gps_results)

    m = folium.Map(
        location=[avg_lat, avg_lon],
        zoom_start=18,
        tiles='Esri.WorldImagery'   # satellite view
    )

    # Add a tile layer label
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/'
              'World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri',
        name='Satellite',
        overlay=False,
    )

    colors = {
        'Healthy':   '#2ecc71',
        'Deficient': '#e74c3c',
        'Uncertain': '#f39c12',
    }
    icons = {
        'Healthy':   '✅',
        'Deficient': '❌',
        'Uncertain': '⚠️',
    }

    for r in gps_results:
        label   = r['classification']
        color   = colors.get(label, '#888888')
        icon    = icons.get(label, '?')
        alt_str = f"{r['alt']} m" if r.get('alt') else 'N/A'

        popup_html = f"""
        <div style="font-family: Arial; min-width: 180px;">
            <h4 style="margin:0 0 6px 0; color:{color}">
                {icon} {label}
            </h4>
            <b>Capture:</b> {r['capture_id']}<br>
            <b>Healthy:</b> {r['healthy_pct']:.1f}%<br>
            <b>Deficient:</b> {r['deficient_pct']:.1f}%<br>
            <b>Margin:</b> {r['margin_pct']:.1f} pp<br>
            <b>Altitude:</b> {alt_str}<br>
            <b>GPS:</b> {r['lat']:.5f}, {r['lon']:.5f}
        </div>
        """

        folium.CircleMarker(
            location=[r['lat'], r['lon']],
            radius=12,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.85,
            popup=folium.Popup(popup_html, max_width=220),
            tooltip=f"{r['capture_id']} — {label}"
        ).add_to(m)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    m.save(output_path)
    return output_path