from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

from pyproj import Transformer
from rasterio.transform import rowcol
from rasterio.windows import Window


CLASS_NAMES = {
    1: "Tree cover",
    2: "Grassland",
    3: "Cropland",
    4: "Built-up",
    5: "Bare/sparse vegetation",
    6: "Permanent water",
}


def create_location_window(
    reference_path,
    latitude,
    longitude,
    area_size_metres=1000,
    pixel_size_metres=10,
):
    """
    Convert latitude/longitude into a centred raster window.

    Returns the selection window and spatial metadata.
    """

    if not -90 <= latitude <= 90:
        raise ValueError("Latitude must be between -90 and 90.")

    if not -180 <= longitude <= 180:
        raise ValueError("Longitude must be between -180 and 180.")

    if area_size_metres <= 0 or pixel_size_metres <= 0:
        raise ValueError("Area and pixel sizes must be positive.")

    if area_size_metres % pixel_size_metres != 0:
        raise ValueError("Area size must be divisible by pixel size.")

    area_pixels = area_size_metres // pixel_size_metres

    with rasterio.open(reference_path) as src:
        transform = src.transform
        crs = src.crs
        height = src.height
        width = src.width
        bounds = src.bounds

        if not np.isclose(abs(transform.a), pixel_size_metres):
            raise ValueError("Unexpected raster pixel width.")

        if not np.isclose(abs(transform.e), pixel_size_metres):
            raise ValueError("Unexpected raster pixel height.")

    transformer = Transformer.from_crs(
        "EPSG:4326",
        crs,
        always_xy=True,
    )

    x, y = transformer.transform(longitude, latitude)

    center_row, center_col = rowcol(transform, x, y)

    half = area_pixels // 2

    row_start = center_row - half
    row_end = row_start + area_pixels

    col_start = center_col - half
    col_end = col_start + area_pixels

    inside_study_area = (
        0 <= center_row < height
        and 0 <= center_col < width
    )

    window_inside = (
        row_start >= 0
        and col_start >= 0
        and row_end <= height
        and col_end <= width
    )

    if not inside_study_area or not window_inside:
        raise ValueError(
            "Invalid location: the centre or complete selection "
            "is outside the study-area raster."
        )

    selection_window = Window(
        col_start,
        row_start,
        area_pixels,
        area_pixels,
    )

    return {
        "window": selection_window,
        "area_pixels": area_pixels,
        "area_size_metres": area_size_metres,
        "latitude": latitude,
        "longitude": longitude,
        "projected_x": x,
        "projected_y": y,
        "center_row": center_row,
        "center_col": center_col,
        "row_start": row_start,
        "row_end": row_end,
        "col_start": col_start,
        "col_end": col_end,
        "crs": crs,
        "transform": transform,
        "height": height,
        "width": width,
        "bounds": bounds,
    }


def extract_monthly_maps(
    prediction_dir,
    output_dir,
    months,
    selection,
    class_names=None,
):
    """
    Extract and save selected-location maps for each month.

    Also calculates monthly class percentages and saves a CSV.
    """

    prediction_dir = Path(prediction_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if class_names is None:
        class_names = CLASS_NAMES

    window = selection["window"]
    area_pixels = selection["area_pixels"]

    monthly_results = []
    selected_maps = {}
    output_paths = {}

    for month in months:

        source_path = prediction_dir / f"lulc_{month}_v5.tif"

        if not source_path.exists():
            raise FileNotFoundError(
                f"Missing monthly prediction: {source_path}"
            )

        with rasterio.open(source_path) as src:

            if (
                src.crs != selection["crs"]
                or src.transform != selection["transform"]
                or src.height != selection["height"]
                or src.width != selection["width"]
            ):
                raise ValueError(
                    f"{month}: Prediction raster does not match "
                    "the reference raster grid."
                )

            selected_map = src.read(1, window=window)

            output_transform = src.window_transform(window)

            profile = src.profile.copy()

        assert selected_map.shape == (area_pixels, area_pixels)

        profile.update(
            height=area_pixels,
            width=area_pixels,
            transform=output_transform,
            compress="LZW",
        )

        output_path = (
            output_dir /
            f"lulc_{month}_selected_1km_v5.tif"
        )

        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(selected_map, 1)

        selected_maps[month] = selected_map
        output_paths[month] = output_path

        total_pixels = selected_map.size
        row = {"month": month}

        for class_id, class_name in class_names.items():

            count = int(np.sum(selected_map == class_id))

            row[class_name] = count / total_pixels * 100

        monthly_results.append(row)

    selected_df = pd.DataFrame(monthly_results)

    csv_path = (
        output_dir /
        "selected_location_monthly_percentages_v5.csv"
    )

    selected_df.to_csv(csv_path, index=False)

    return {
        "maps": selected_maps,
        "statistics": selected_df,
        "output_paths": output_paths,
        "csv_path": csv_path,
        "output_dir": output_dir,
    }
from pyproj import CRS, Transformer
from affine import Affine


def create_utm_grid(
    latitude,
    longitude,
    area_size_m=1000,
    pixel_size_m=10
):
    """
    Create a square UTM grid centred on a geographic coordinate.

    Returns grid metadata required for GEE export and GeoTIFF creation.
    """

    if not -80 <= latitude <= 84:
        raise ValueError(
            "Latitude must be between -80 and 84 for UTM."
        )

    if not -180 <= longitude <= 180:
        raise ValueError("Invalid longitude.")

    if area_size_m <= 0 or pixel_size_m <= 0:
        raise ValueError("Area and pixel sizes must be positive.")

    if area_size_m % pixel_size_m != 0:
        raise ValueError(
            "Area size must be divisible by pixel size."
        )

    zone = int((longitude + 180) // 6) + 1
    zone = min(max(zone, 1), 60)

    epsg = (
        32600 + zone if latitude >= 0
        else 32700 + zone
    )

    crs = CRS.from_epsg(epsg)

    transformer = Transformer.from_crs(
        "EPSG:4326",
        crs,
        always_xy=True
    )

    x, y = transformer.transform(longitude, latitude)

    half = area_size_m / 2

    left = x - half
    right = x + half
    bottom = y - half
    top = y + half

    width = area_size_m // pixel_size_m
    height = width

    transform = Affine(
        pixel_size_m,
        0,
        left,
        0,
        -pixel_size_m,
        top
    )

    return {
        "crs": crs.to_string(),
        "epsg": epsg,
        "projection": ee_projection_placeholder if False else None,
        "transform": [
            pixel_size_m, 0, left,
            0, -pixel_size_m, top
        ],
        "affine": transform,
        "bounds": (left, bottom, right, top),
        "width": width,
        "height": height,
        "center_x": x,
        "center_y": y,
        "latitude": latitude,
        "longitude": longitude,
        "area_size_m": area_size_m,
        "pixel_size_m": pixel_size_m
    }