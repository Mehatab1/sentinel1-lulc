from pathlib import Path
from collections import Counter

import numpy as np
import rasterio
import torch

from .model import load_v5_model


CLASS_NAMES = {
    1: "Tree cover",
    2: "Grassland",
    3: "Cropland",
    4: "Built-up",
    5: "Bare/sparse vegetation",
    6: "Permanent water",
}


def predict_raster(
    input_path,
    output_path,
    checkpoint_path,
    batch_size=16,
    patch_size=128,
    device=None,
):
    """
    Generate a Sentinel-1 LULC prediction GeoTIFF.

    Preprocessing and tiling follow the original V5 prototype:
    - VV/VH checkpoint-based median-IQR normalization
    - 128x128 non-overlapping tiles
    - Reflect padding at bottom and right edges
    - Batched U-Net inference
    - Argmax predictions converted to class IDs 1-6
    """

    input_path = Path(input_path)
    output_path = Path(output_path)
    checkpoint_path = Path(checkpoint_path)

    if not input_path.is_file():
        raise FileNotFoundError(input_path)

    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)

    if batch_size < 1 or patch_size < 1:
        raise ValueError("batch_size and patch_size must be positive.")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load the trained model and checkpoint metadata.
    model, checkpoint = load_v5_model(
        checkpoint_path,
        device=device,
    )

    device = next(model.parameters()).device

    if checkpoint["architecture"] != "UNet":
        raise ValueError("Unexpected checkpoint architecture.")

    if checkpoint["in_channels"] != 2:
        raise ValueError("Expected two Sentinel-1 input bands.")

    if checkpoint["num_classes"] != 6:
        raise ValueError("Expected six output classes.")

    medians = np.asarray(
        checkpoint["class_medians"], dtype=np.float32
    )
    iqr = np.asarray(
        checkpoint["class_iqr"], dtype=np.float32
    )

    if np.any(iqr == 0):
        raise ValueError("Invalid zero IQR normalization value.")

    # Read radar raster.
    with rasterio.open(input_path) as src:
        if src.count != 2:
            raise ValueError("Input must contain VV and VH bands.")

        radar = src.read().astype(np.float32)
        profile = src.profile.copy()
        height, width = src.height, src.width
        nodata = src.nodata

    if radar.shape != (2, height, width):
        raise ValueError("Unexpected radar array dimensions.")

    valid_mask = np.isfinite(radar).all(axis=0)

    if nodata is not None:
        valid_mask &= (radar != nodata).all(axis=0)

    if not valid_mask.all():
        raise ValueError(
            "Input contains invalid or NoData pixels. "
            "The original V5 inference requires complete coverage."
        )

    # Normalize VV and VH independently using training statistics.
    for band in range(2):
        radar[band] = (
            radar[band] - medians[band]
        ) / iqr[band]

    if not np.isfinite(radar).all():
        raise ValueError("Non-finite values after normalization.")

    # Pad dimensions to multiples of patch_size.
    padded_height = (
        (height + patch_size - 1) // patch_size
    ) * patch_size

    padded_width = (
        (width + patch_size - 1) // patch_size
    ) * patch_size

    pad_bottom = padded_height - height
    pad_right = padded_width - width

    radar_padded = np.pad(
        radar,
        ((0, 0), (0, pad_bottom), (0, pad_right)),
        mode="reflect",
    )

    # Generate tile locations in the original row-major order.
    locations = [
        (y, x)
        for y in range(0, padded_height, patch_size)
        for x in range(0, padded_width, patch_size)
    ]

    prediction_padded = np.zeros(
        (padded_height, padded_width),
        dtype=np.uint8,
    )

    model.eval()

    # Batched inference.
    with torch.no_grad():
        for start in range(0, len(locations), batch_size):

            batch_locations = locations[start:start + batch_size]

            batch_patches = np.stack([
                radar_padded[
                    :,
                    y:y + patch_size,
                    x:x + patch_size,
                ]
                for y, x in batch_locations
            ])

            batch_tensor = torch.from_numpy(
                batch_patches
            ).to(device)

            logits = model(batch_tensor)

            predicted_classes = (
                torch.argmax(logits, dim=1)
                .cpu()
                .numpy()
                .astype(np.uint8)
            )

            for (y, x), predicted_patch in zip(
                batch_locations, predicted_classes
            ):
                prediction_padded[
                    y:y + patch_size,
                    x:x + patch_size,
                ] = predicted_patch + 1

    # Remove padding and validate output IDs.
    prediction = prediction_padded[:height, :width]

    if prediction.shape != (height, width):
        raise RuntimeError("Unexpected prediction dimensions.")

    if not set(np.unique(prediction)).issubset(
        {1, 2, 3, 4, 5, 6}
    ):
        raise RuntimeError("Unexpected predicted class IDs.")

    # Save georeferenced prediction.
    profile.update(
        driver="GTiff",
        count=1,
        dtype="uint8",
        nodata=0,
        compress="LZW",
    )

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(prediction, 1)

    # Summarize class distribution.
    counts = Counter(prediction.ravel())
    total_pixels = prediction.size

    class_summary = {
        class_id: {
            "name": class_name,
            "pixels": counts.get(class_id, 0),
            "percentage": (
                counts.get(class_id, 0) / total_pixels * 100
            ),
        }
        for class_id, class_name in CLASS_NAMES.items()
    }

    return {
        "input": str(input_path),
        "output": str(output_path),
        "shape": prediction.shape,
        "total_pixels": total_pixels,
        "class_distribution": class_summary,
        "padding": {
            "bottom": pad_bottom,
            "right": pad_right,
        },
        "device": str(device),
    }
