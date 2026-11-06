from pathlib import Path

import numpy as np
import pandas as pd


CLASS_NAMES = {
    1: "Tree cover",
    2: "Grassland",
    3: "Cropland",
    4: "Built-up",
    5: "Bare/sparse vegetation",
    6: "Permanent water",
}

CLASS_IDS = list(CLASS_NAMES.keys())


def calculate_transition_matrix(previous_map, current_map):
    """Calculate a 6x6 class-to-class transition matrix."""

    if previous_map.shape != current_map.shape:
        raise ValueError("Input maps must have identical shapes.")

    matrix = np.zeros((6, 6), dtype=np.int64)

    for i, from_class in enumerate(CLASS_IDS):
        for j, to_class in enumerate(CLASS_IDS):
            matrix[i, j] = np.sum(
                (previous_map == from_class)
                & (current_map == to_class)
            )

    return matrix


def analyze_monthly_maps(
    monthly_maps,
    months,
    output_dir=None,
    pixel_size_m=10,
):

    if pixel_size_m <= 0:
        raise ValueError("pixel_size_m must be greater than zero.")

    pixel_area_ha = (pixel_size_m ** 2) / 10000


    """
    Calculate monthly percentages, temporal stability,
    switches, month-to-month changes and transition matrices.

    monthly_maps: dictionary mapping month strings to 2D arrays.
    months: ordered list of month strings.
    """

    if len(months) < 2:
        raise ValueError("At least two months are required.")

    if len(set(months)) != len(months):
        raise ValueError("Month names must be unique.")

    missing = [m for m in months if m not in monthly_maps]

    if missing:
        raise ValueError(f"Missing monthly maps: {missing}")

    reference_shape = monthly_maps[months[0]].shape

    if len(reference_shape) != 2:
        raise ValueError("Monthly maps must be 2D arrays.")

    for month in months:
        if monthly_maps[month].shape != reference_shape:
            raise ValueError(f"{month}: inconsistent map dimensions.")

        values = np.unique(monthly_maps[month])
        if not np.isin(values, CLASS_IDS).all():
            raise ValueError(
                f"{month}: unexpected class IDs {values.tolist()}"
            )

    stack = np.stack(
        [monthly_maps[m] for m in months],
        axis=0,
    )

    total_pixels = stack.shape[1] * stack.shape[2]

    # --------------------------------------------------
    # 1. Monthly class percentages
    # --------------------------------------------------

    monthly_rows = []

    for month in months:
        current_map = monthly_maps[month]

        row = {"month": month}

        for class_id, class_name in CLASS_NAMES.items():
            count = int(np.sum(current_map == class_id))
            row[class_name] = count / total_pixels * 100

        monthly_rows.append(row)

    monthly_df = pd.DataFrame(monthly_rows)

    # --------------------------------------------------
    # 2. First-to-last transition analysis
    # --------------------------------------------------

    first_map = monthly_maps[months[0]]
    last_map = monthly_maps[months[-1]]

    changed_mask = first_map != last_map
    changed_pixels = int(changed_mask.sum())
    unchanged_pixels = total_pixels - changed_pixels

    first_last_matrix = calculate_transition_matrix(
        first_map,
        last_map,
    )

    first_last_df = pd.DataFrame(
        first_last_matrix,
        index=list(CLASS_NAMES.values()),
        columns=list(CLASS_NAMES.values()),
    )

    first_last_df.index.name = "From / To"

    # --------------------------------------------------
    # 3. Temporal stability
    # --------------------------------------------------

    unique_class_count = np.zeros(
        reference_shape,
        dtype=np.uint8,
    )

    for r in range(reference_shape[0]):
        for c in range(reference_shape[1]):
            unique_class_count[r, c] = len(
                np.unique(stack[:, r, c])
            )

    stability = {
        "stable_all_months": int(
            np.sum(unique_class_count == 1)
        ),
        "two_distinct_classes": int(
            np.sum(unique_class_count == 2)
        ),
        "three_or_more_distinct_classes": int(
            np.sum(unique_class_count >= 3)
        ),
    }

    # --------------------------------------------------
    # 4. Consecutive-month switches
    # --------------------------------------------------

    monthly_switches = np.sum(
        stack[1:] != stack[:-1],
        axis=0,
    )

    switch_distribution = []

    for n in range(len(months)):
        count = int(np.sum(monthly_switches == n))

        switch_distribution.append({
            "switches": n,
            "pixels": count,
            "percentage": count / total_pixels * 100,
        })

    switch_df = pd.DataFrame(switch_distribution)

    # --------------------------------------------------
    # 5. Month-to-month changed pixels
    # --------------------------------------------------

    change_rows = []

    for i in range(1, len(months)):
        previous_map = stack[i - 1]
        current_map = stack[i]

        changed = previous_map != current_map
        count = int(changed.sum())

        change_rows.append({
            "period": f"{months[i-1]} → {months[i]}",
            "changed_pixels": count,
            "changed_area_hectares": count * pixel_area_ha,
            "area_hectares": count * pixel_area_ha,
        })

    monthly_change_df = pd.DataFrame(change_rows)

    # --------------------------------------------------
    # 6. First-to-last class transitions
    # --------------------------------------------------

    transition_rows = []

    for i, from_class in enumerate(CLASS_IDS):
        for j, to_class in enumerate(CLASS_IDS):

            count = int(first_last_matrix[i, j])

            if count > 0:
                transition_rows.append({
                    "from_class": CLASS_NAMES[from_class],
                    "to_class": CLASS_NAMES[to_class],
                    "pixels": count,
                    "area_hectares": count * 0.01,
                })

    transition_details_df = pd.DataFrame(transition_rows)

    # --------------------------------------------------
    # 7. Save outputs
    # --------------------------------------------------

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        monthly_df.to_csv(
            output_dir / "monthly_percentages.csv",
            index=False,
        )

        first_last_df.to_csv(
            output_dir / "first_last_transition_matrix.csv",
        )

        switch_df.to_csv(
            output_dir / "switch_distribution.csv",
            index=False,
        )

        monthly_change_df.to_csv(
            output_dir / "month_to_month_changes.csv",
            index=False,
        )

        transition_details_df.to_csv(
            output_dir / "first_last_transition_details.csv",
            index=False,
        )

    return {
        "monthly_percentages": monthly_df,
        "first_last_transition_matrix": first_last_df,
        "changed_pixels": changed_pixels,
        "unchanged_pixels": unchanged_pixels,
        "changed_percentage": changed_pixels / total_pixels * 100,
        "unique_class_count": unique_class_count,
        "stability": stability,
        "monthly_switches": monthly_switches,
        "switch_distribution": switch_df,
        "monthly_changes": monthly_change_df,
        "transition_details": transition_details_df,
        "total_pixels": total_pixels,
    }


def calculate_interval_transitions(
    monthly_maps,
    start_month,
    end_month,
):
    """Calculate class transitions between two selected months."""

    if start_month not in monthly_maps:
        raise KeyError(f"Missing month: {start_month}")

    if end_month not in monthly_maps:
        raise KeyError(f"Missing month: {end_month}")

    matrix = calculate_transition_matrix(
        monthly_maps[start_month],
        monthly_maps[end_month],
    )

    result = pd.DataFrame(
        matrix,
        index=list(CLASS_NAMES.values()),
        columns=list(CLASS_NAMES.values()),
    )

    result.index.name = "From / To"

    return result
