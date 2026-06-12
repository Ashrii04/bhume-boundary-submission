"""My boundary corrector."""

import numpy as np
import cv2
from shapely.geometry import shape, Point
from shapely.affinity import translate
import geopandas as gpd

from bhume import load, patch_for_plot, write_predictions, score
from bhume.geo import open_imagery, geom_to_imagery_crs


def detect_edges(patch_image):
    """Turn an RGB image patch into a black/white edge map."""
    gray = cv2.cvtColor(patch_image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    return edges
def plot_outline_mask(geom_4326, src, patch):
    """Draw the plot's OUTLINE (just the border) as a mask matching the patch image."""
    h, w = patch.image.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    # Reproject the plot polygon into the same CRS as the imagery
    geom_img_crs = geom_to_imagery_crs(src, geom_4326)

    # Get the polygon's boundary points, convert each to pixel (col, row)
    left, bottom, right, top = patch.bounds
    transform = patch.transform

   # Handle both Polygon and MultiPolygon
    if geom_img_crs.geom_type == "Polygon":
        polys = [geom_img_crs]
    else:  # MultiPolygon
        polys = list(geom_img_crs.geoms)

    for poly in polys:
        coords = list(poly.exterior.coords)
        pts = []
        for x, y in coords:
            col, row = ~transform * (x, y)
            pts.append((int(col), int(row)))
        pts = np.array([pts], dtype=np.int32)
        cv2.polylines(mask, pts, isClosed=True, color=255, thickness=2)

    return mask

def find_best_shift(outline_mask, edges, max_shift_px=15):
    """Slide the outline over the edge map; find the offset with best overlap."""
    h, w = outline_mask.shape
    best_score = -1.0
    best_dx, best_dy = 0, 0

    outline_pts = np.sum(outline_mask > 0)
    if outline_pts == 0:
        return 0, 0, 0.0

    # Dilate edges slightly so "close enough" counts as a match
    edges_dilated = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    for dy in range(-max_shift_px, max_shift_px + 1, 1):
        for dx in range(-max_shift_px, max_shift_px + 1, 1):
            shifted = np.roll(outline_mask, (dy, dx), axis=(0, 1))
            overlap = np.sum((shifted > 0) & (edges_dilated > 0))
            score = overlap / outline_pts
            if score > best_score:
                best_score = score
                best_dx, best_dy = dx, dy

    return best_dx, best_dy, best_score
def shift_geometry(geom_4326, src, patch, dx_px, dy_px):
    """Apply a pixel shift to a lon/lat geometry, returning a new lon/lat geometry."""
    transform = patch.transform

    # pixel size in the imagery CRS (metres per pixel)
    px_size_x = transform.a   # width of one pixel
    px_size_y = transform.e   # height of one pixel (usually negative)

    # convert pixel shift to imagery-CRS shift (metres)
    dx_m = dx_px * px_size_x
    dy_m = dy_px * px_size_y   # note: row increases downward, y increases upward

    # work in imagery CRS, then convert back
    geom_img_crs = geom_to_imagery_crs(src, geom_4326)
    shifted_img = translate(geom_img_crs, xoff=dx_m, yoff=dy_m)

    # reproject back to lon/lat
    from pyproj import Transformer
    from shapely.ops import transform as shp_transform
    tf = Transformer.from_crs(src.crs, 'EPSG:4326', always_xy=True)
    shifted_4326 = shp_transform(lambda xs, ys, z=None: tf.transform(xs, ys), shifted_img)

    return shifted_4326
def process_village(village_name):
    village = load(f"data/{village_name}")

    results = []
    with open_imagery(village.imagery_path) as src:
        for pn in village.plots.index:
            geom = village.plot(pn)
            row = village.plots.loc[pn]

            try:
                patch = patch_for_plot(src, geom, pad_m=30)
                edges = detect_edges(patch.image)
                outline = plot_outline_mask(geom, src, patch)
                dx, dy, align_score = find_best_shift(outline, edges, max_shift_px=15)
            except Exception as e:
                print(f"PLOT {pn} ERROR: {repr(e)}")
                results.append({
                    "plot_number": pn,
                    "status": "flagged",
                    "confidence": None,
                    "method_note": "error reading patch",
                    "geometry": geom,
                })
                continue

            # --- area ratio signal ---
            recorded = row.get("recorded_area_sqm") or 0
            pot_kharaba_sqm = (row.get("pot_kharaba_ha") or 0) * 10000
            total_recorded = recorded + pot_kharaba_sqm
            mapped = row.get("map_area_sqm") or 0
            area_ratio = (mapped / total_recorded) if total_recorded > 0 else 1.0

            # --- combine into confidence ---
            confidence = align_score
            if 0.85 < area_ratio < 1.15:
                confidence *= 1.1
            elif area_ratio < 0.6 or area_ratio > 1.4:
                confidence *= 0.6

            shift_dist_px = (dx**2 + dy**2) ** 0.5
            if shift_dist_px > 10:
                confidence *= 0.7

            confidence = round(min(1.0, max(0.0, confidence)), 2)

            # --- decide ---
            if confidence >= 0.5 and align_score > 0.3:
                new_geom = shift_geometry(geom, src, patch, dx, dy)
                if not new_geom.is_valid or new_geom.is_empty:
                    status, conf, out_geom = "flagged", None, geom
                    note = "invalid geometry after shift"
                else:
                    status, conf, out_geom = "corrected", confidence, new_geom
                    note = f"shift=({dx},{dy})px align={align_score:.2f} area_ratio={area_ratio:.2f}"
            else:
                status, conf, out_geom = "flagged", None, geom
                note = f"low confidence align={align_score:.2f} area_ratio={area_ratio:.2f}"

            results.append({
                "plot_number": pn,
                "status": status,
                "confidence": conf,
                "method_note": note,
                "geometry": out_geom,
            })

    gdf = gpd.GeoDataFrame(results, geometry="geometry", crs="EPSG:4326")
    out_path = village.dir / "predictions.geojson"
    write_predictions(out_path, gdf)
    print(f"wrote {len(gdf)} predictions -> {out_path}")

    sc = score(gdf, village)
    print(sc)
    return gdf


if __name__ == "__main__":
    import sys
    village_name = sys.argv[1] if len(sys.argv) > 1 else "vadnerbhairav"
    process_village(village_name)