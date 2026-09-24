"""Small geodesy helpers (equirectangular projection is plenty at corridor scale)."""

from __future__ import annotations

import numpy as np

EARTH_RADIUS_M = 6_371_000.0
# Projection reference latitude: middle of the Boston-DC corridor. At this
# scale the east-west distortion across the corridor stays under ~3%, well
# inside the slack of walking-distance estimates.
REF_LAT = 40.6


def project(lat, lon) -> np.ndarray:
    """Project lat/lon (degrees) to planar metres, shape (n, 2)."""
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    x = np.radians(lon) * EARTH_RADIUS_M * np.cos(np.radians(REF_LAT))
    y = np.radians(lat) * EARTH_RADIUS_M
    return np.column_stack([x, y])


def haversine_m(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = (np.radians(np.asarray(v, dtype=float)) for v in (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(a))


def distance_to_polyline_m(lat, lon, spine) -> np.ndarray:
    """Planar distance from each point to the nearest segment of `spine`."""
    pts = project(lat, lon)
    sp = project([p[0] for p in spine], [p[1] for p in spine])
    best = np.full(len(pts), np.inf)
    for a, b in zip(sp[:-1], sp[1:]):
        ab = b - a
        t = np.clip(((pts - a) @ ab) / (ab @ ab), 0.0, 1.0)
        proj = a + t[:, None] * ab
        best = np.minimum(best, np.linalg.norm(pts - proj, axis=1))
    return best


def bbox_near_polyline(min_lat, max_lat, min_lon, max_lon, spine, buffer_m) -> bool:
    """True if the bbox comes within buffer_m of the polyline.

    Tests the bbox corners/edge midpoints against the spine and samples of the
    spine against the bbox, which is exact enough for feed selection.
    """
    lats = np.linspace(min_lat, max_lat, 5)
    lons = np.linspace(min_lon, max_lon, 5)
    glat, glon = np.meshgrid(lats, lons)
    if distance_to_polyline_m(glat.ravel(), glon.ravel(), spine).min() <= buffer_m:
        return True
    # Spine passing through a large bbox without any grid point being close.
    for (la, lo), (lb, lob) in zip(spine[:-1], spine[1:]):
        for t in np.linspace(0, 1, 20):
            plat, plon = la + t * (lb - la), lo + t * (lob - lo)
            if min_lat <= plat <= max_lat and min_lon <= plon <= max_lon:
                return True
    return False


def unproject(xy) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of project(): planar metres -> (lat, lon) degrees."""
    xy = np.atleast_2d(np.asarray(xy, dtype=float))
    lat = np.degrees(xy[:, 1] / EARTH_RADIUS_M)
    lon = np.degrees(xy[:, 0] / (EARTH_RADIUS_M * np.cos(np.radians(REF_LAT))))
    return lat, lon
