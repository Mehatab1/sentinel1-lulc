# Sentinel-1-Based Land Use and Land Cover (LULC) Prototype

## 1. Project Overview

This project implements a deep-learning-based semantic segmentation
prototype for Land Use and Land Cover (LULC) mapping using Sentinel-1
VV and VH radar imagery.

The trained model generates monthly LULC predictions and calculates
class distribution, temporal differences, and spatial transitions.

The current prototype uses pre-generated monthly Stuttgart prediction
maps and supports selection of a 1 km x 1 km area within the study area.

## 2. Study Area and Observation Period

- Study area: Stuttgart, Germany
- Original study extent: approximately 20 km x 20 km
- Coordinate reference system: EPSG:32632
- Spatial resolution: 10 metres
- Observation period: October 2025 – September 2026
- Monthly observations: 12

## 3. Data Sources

### Sentinel-1

- Dataset: COPERNICUS/S1_GRD
- Instrument mode: IW
- Orbit direction: Descending
- Polarizations: VV and VH
- Monthly compositing: Median
- Spatial resolution: 10 metres

### Reference Labels

- Dataset: ESA WorldCover v200
- Reference year: 2021
- Original classification labels remapped to six classes.

## 4. LULC Classes

| Model ID | Class |
|---|---|
| 1 | Tree cover |
| 2 | Grassland |
| 3 | Cropland |
| 4 | Built-up |
| 5 | Bare/sparse vegetation |
| 6 | Permanent water |

Wetland class 90 was excluded from model training.

## 5. Deep Learning Model

- Architecture: U-Net
- Input channels: 2 (VV and VH)
- Output classes: 6
- Base channels: 16
- Framework: PyTorch
- Optimizer: AdamW
- Learning rate: 0.001
- Weight decay: 0.0001
- Batch size: 4
- Maximum epochs: 30

### V5 Training

The V5 training dataset combines the original training locations
with additional training patches from an expanded geographic area.

- Original training locations: 90
- Additional outer-ring locations: 600
- Unique training locations: 690
- Monthly observations per location: 12

Validation and test locations were retained from the original
spatial split.

### V5 Evaluation

| Metric | Result |
|---|---:|
| Test loss | 0.7013 |
| Overall accuracy | 0.7190 |
| Mean IoU | 0.4734 |

Bare/sparse vegetation was not reliably detected.

## 6. Prototype Workflow

1. Load the trained V5 checkpoint.
2. Load the 12 monthly Sentinel-1 prediction inputs.
3. Apply the training normalization parameters.
4. Generate monthly LULC predictions.
5. Save monthly prediction maps as GeoTIFF files.
6. Calculate class percentages and area statistics.
7. Calculate month-to-month changes.
8. Compare the first and last monthly predictions.
9. Generate transition matrices and temporal stability maps.
10. Select a 1 km x 1 km location from the existing prediction maps.

## 7. Location-Based Analysis

The current prototype supports selection of a 1 km x 1 km window
from the existing Stuttgart prediction maps using latitude and
longitude.

It generates:

- 12 monthly local LULC maps
- Monthly class percentages
- Monthly change statistics
- First-to-last change map
- Class transition matrix
- Temporal stability map
- Consecutive-month classification switch analysis

Important: this implementation extracts from existing prediction
rasters. It does not yet support arbitrary geographic locations or
automatic live Sentinel-1 acquisition.

## 8. Current Local Analysis Results

Selected area: 1 km x 1 km

Observation period: October 2025 – September 2026

| Metric | Result |
|---|---:|
| Pixels | 10,000 |
| Stable across all 12 months | 86.92% |
| Pixels changing at least once | 13.08% |
| Mean consecutive-month switches | 0.459 |
| October-to-September changed pixels | 3.89% |
| Highest monthly variability | 5.95% |

These figures describe model predictions, not independently
verified physical land-cover changes.

## 9. Outputs

The prototype generates:

- Monthly LULC GeoTIFF maps
- Monthly class percentage CSV
- Monthly change CSV
- First-to-last comparison CSV
- Class transition matrices
- Spatial change maps
- Temporal stability visualizations
- Monthly trend charts
- 12-month LULC animation

Selected-location outputs are preserved in:

`deliverables_v5/local_location_analysis/`

## 10. Scientific Limitations

1. WorldCover 2021 is used as static reference data for radar
   observations from 2025–2026.
2. The reference labels are not monthly ground truth.
3. WorldCover incorporates Sentinel-1 and Sentinel-2 information,
   so it is not fully independent of the model input modality.
4. Bare/sparse vegetation has weak model performance.
5. Seasonal radar responses can influence monthly predictions.
6. Predicted class transitions do not establish verified
   real-world land-cover conversions.
7. The current location selector is restricted to the existing
   Stuttgart prediction coverage.

## 11. Future Improvements

- Support arbitrary geographic locations.
- Integrate automatic Sentinel-1 data acquisition.
- Improve rare-class representation and performance.
- Evaluate against independent and temporally appropriate
  reference data.
- Add confidence and uncertainty visualization.
- Develop a user-friendly interactive interface.

## 12. Reproducibility

The trained V5 checkpoint, monthly input rasters, notebooks,
normalization parameters, and generated outputs should be retained
together with the project.

Refer to the project notebooks for the detailed data preparation,
training, evaluation, and inference procedures.
