import ee

def create_ee_grid(grid):
    """
    Convert a UTM grid dictionary into an Earth Engine
    projection and rectangular region.
    """

    # Projection with the exact pixel alignment
    projection = ee.Projection(
        grid["crs"],
        grid["transform"],
    )

    # Region uses the CRS only, without the affine transform
    left, bottom, right, top = grid["bounds"]

    region = ee.Geometry.Rectangle(
        [left, bottom, right, top],
        proj=grid["crs"],
        geodesic=False,
    )

    return {
        "projection": projection,
        "region": region,
    }

def get_sentinel1_collection(region, start_date, end_date):
    """
    Retrieve Sentinel-1 GRD scenes for a region and date range.

    Uses IW mode, descending pass, and VV/VH polarization.
    """

    collection = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(region)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.eq("orbitProperties_pass", "DESCENDING"))
        .filter(ee.Filter.listContains(
            "transmitterReceiverPolarisation", "VV"
        ))
        .filter(ee.Filter.listContains(
            "transmitterReceiverPolarisation", "VH"
        ))
        .select(["VV", "VH"])
    )

    return collection
def get_monthly_scene_counts(region, months):
    """
    Return Sentinel-1 scene counts for each requested month.

    months: list of dictionaries with:
        {"label": "2025-10", "start": "2025-10-01",
         "end": "2025-11-01"}
    """

    results = []

    for month in months:
        collection = get_sentinel1_collection(
            region,
            month["start"],
            month["end"],
        )

        scene_count = collection.size().getInfo()

        results.append({
            "month": month["label"],
            "scene_count": scene_count,
            "available": scene_count > 0,
        })

    return results

def get_monthly_orbit_counts(region, months):
    """
    Return scene counts grouped by relative orbit for each month.
    """

    results = []

    for month in months:
        collection = get_sentinel1_collection(
            region,
            month["start"],
            month["end"],
        )

        orbit_counts = collection.aggregate_histogram(
            "relativeOrbitNumber_start"
        ).getInfo()

        results.append({
            "month": month["label"],
            "orbit_counts": {
                int(orbit): int(count)
                for orbit, count in orbit_counts.items()
            },
        })

    return results
def select_consistent_orbit(orbit_results):
    """
    Select the relative orbit with the highest total scene count
    among orbits available in every requested month.

    Returns the selected orbit and availability details.
    """

    if not orbit_results:
        raise ValueError("No monthly orbit results provided.")

    common_orbits = set(orbit_results[0]["orbit_counts"])

    for item in orbit_results[1:]:
        common_orbits &= set(item["orbit_counts"])

    if not common_orbits:
        return {
            "selected_orbit": None,
            "consistent": False,
            "common_orbits": [],
        }

    totals = {
        orbit: sum(
            item["orbit_counts"].get(orbit, 0)
            for item in orbit_results
        )
        for orbit in common_orbits
    }

    selected_orbit = max(totals, key=totals.get)

    return {
        "selected_orbit": selected_orbit,
        "consistent": True,
        "common_orbits": sorted(common_orbits),
        "scene_counts": totals,
    }

def create_monthly_composite(
    region,
    projection,
    month,
    relative_orbit,
):
    """
    Create a monthly Sentinel-1 VV/VH median composite
    using a selected relative orbit.
    """

    collection = (
        get_sentinel1_collection(
            region,
            month["start"],
            month["end"],
        )
        .filter(
            ee.Filter.eq(
                "relativeOrbitNumber_start",
                relative_orbit,
            )
        )
    )

    scene_count = collection.size().getInfo()

    if scene_count == 0:
        raise ValueError(
            f"No Sentinel-1 scenes for {month['label']} "
            f"on relative orbit {relative_orbit}."
        )

    composite = (
        collection
        .median()
        .select(["VV", "VH"])
        .setDefaultProjection(projection)
        .clip(region)
        .set({
            "month": month["label"],
            "relative_orbit": relative_orbit,
            "scene_count": scene_count,
        })
    )

    return composite
from pathlib import Path
import requests
import zipfile
import io


def download_monthly_composite(
    composite,
    region,
    grid,
    output_path,
):
    """
    Download a two-band Sentinel-1 composite as GeoTIFF.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    download_url = composite.getDownloadURL({
        "name": output_path.stem,
        "crs": grid["crs"],
        "crs_transform": grid["transform"],
        "region": region,
        "format": "GEO_TIFF",
    })

    response = requests.get(download_url, timeout=180)
    response.raise_for_status()

    content = response.content

    # Earth Engine may return a ZIP containing the GeoTIFF.
    if zipfile.is_zipfile(io.BytesIO(content)):
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            tif_files = [
                name for name in archive.namelist()
                if name.lower().endswith((".tif", ".tiff"))
            ]

            if not tif_files:
                raise ValueError("No GeoTIFF found in download archive.")

            with archive.open(tif_files[0]) as source:
                output_path.write_bytes(source.read())
    else:
        output_path.write_bytes(content)

    return output_path