
import sys
from pathlib import Path

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import plotly.express as px

# --------------------------------------------------
# PROJECT PATHS AND IMPORTS
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))

from src.location import (
    create_location_window,
    extract_monthly_maps,
    CLASS_NAMES
)

from src.analysis import analyze_monthly_maps

PREDICTION_DIR = PROJECT_DIR / "prototype_outputs"
OUTPUT_DIR = PREDICTION_DIR / "dashboard_results"

MONTHS = [
    "2025_10", "2025_11", "2025_12", "2026_01",
    "2026_02", "2026_03", "2026_04", "2026_05",
    "2026_06", "2026_07", "2026_08", "2026_09"
]

# --------------------------------------------------
# Shared LULC class colours
# Keep these colours consistent across all dashboard tabs.
# --------------------------------------------------

LULC_COLORS = {
    "Tree cover": "#26734D",
    "Grassland": "#A8C66C",
    "Cropland": "#E6C75A",
    "Built-up": "#D95F59",
    "Bare/sparse vegetation": "#B99B82",
    "Permanent water": "#4A90C2",
}

LULC_CLASS_IDS = {
    1: "Tree cover",
    2: "Grassland",
    3: "Cropland",
    4: "Built-up",
    5: "Bare/sparse vegetation",
    6: "Permanent water",
}


MONTH_LABELS = {
    "2025_10": "October 2025",
    "2025_11": "November 2025",
    "2025_12": "December 2025",
    "2026_01": "January 2026",
    "2026_02": "February 2026",
    "2026_03": "March 2026",
    "2026_04": "April 2026",
    "2026_05": "May 2026",
    "2026_06": "June 2026",
    "2026_07": "July 2026",
    "2026_08": "August 2026",
    "2026_09": "September 2026"
}

CLASS_COLORS = [
    "#26734D",  # Tree cover
    "#A8C66C",  # Grassland
    "#E6C35C",  # Cropland
    "#D65A4A",  # Built-up
    "#B9A18A",  # Bare/sparse
    "#4A90C2"   # Permanent water
]

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Sentinel-1 LULC Explorer",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.block-container {
    padding-top: 1.8rem;
    padding-bottom: 2rem;
}
.main-title {
    font-size: 2.1rem;
    font-weight: 750;
    letter-spacing: -1px;
    margin-bottom: 0.2rem;
}
.subtitle {
    color: #8492A6;
    font-size: 1rem;
    margin-bottom: 1.5rem;
}
.section-title {
    font-size: 1.2rem;
    font-weight: 650;
    margin-top: 1rem;
}
.status-note {
    color: #8492A6;
    font-size: 0.85rem;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown(
    '<div class="main-title">🌍 Sentinel-1 LULC Explorer</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Satellite-based Land Use and Land Cover monitoring '
    'using deep learning'
    '</div>',
    unsafe_allow_html=True
)

st.divider()

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.title("📍 Study Area")

st.sidebar.markdown(
    "Select a location within the current Stuttgart study area."
)

latitude = st.sidebar.number_input(
    "Latitude",
    min_value=-90.0,
    max_value=90.0,
    value=48.7758,
    format="%.6f"
)

longitude = st.sidebar.number_input(
    "Longitude",
    min_value=-180.0,
    max_value=180.0,
    value=9.1829,
    format="%.6f"
)

area_size = st.sidebar.selectbox(
    "Selection area",
    options=[1000],
    format_func=lambda x: "1 km × 1 km"
)

start_month = st.sidebar.selectbox(
    "Starting month",
    options=["2025_10"],
    format_func=lambda x: MONTH_LABELS[x]
)

st.sidebar.caption(
    "Available complete sequence: October 2025 – September 2026"
)

generate = st.sidebar.button(
    "🚀 Generate LULC Analysis",
    type="primary",
    use_container_width=True
)

# --------------------------------------------------
# STUDY AREA OVERVIEW
# --------------------------------------------------

st.subheader("Study Area Overview")

metric1, metric2, metric3, metric4 = st.columns(4)

metric1.metric("Study area", "1 km²")
metric2.metric("Time series", "12 months")
metric3.metric("Input data", "Sentinel-1 VV/VH")
metric4.metric("Model", "U-Net V5")

st.divider()

# --------------------------------------------------
# GENERATE ANALYSIS
# --------------------------------------------------

if generate:

    try:
        with st.spinner("Validating location and processing monthly maps..."):

            reference_path = (
                PREDICTION_DIR / "lulc_2025_10_v5.tif"
            )

            if not reference_path.exists():
                raise FileNotFoundError(
                    f"Reference prediction not found: {reference_path}"
                )

            selection = create_location_window(
                reference_path=reference_path,
                latitude=latitude,
                longitude=longitude,
                area_size_metres=area_size
            )

            extraction = extract_monthly_maps(
                prediction_dir=PREDICTION_DIR,
                output_dir=OUTPUT_DIR / "maps",
                months=MONTHS,
                selection=selection,
                class_names=CLASS_NAMES
            )

            analysis = analyze_monthly_maps(
                monthly_maps=extraction["maps"],
                months=MONTHS,
                output_dir=OUTPUT_DIR / "analysis"
            )

            st.session_state["lulc_results"] = {
                "latitude": latitude,
                "longitude": longitude,
                "selection": selection,
                "months": MONTHS,
                "extraction": extraction,
                "analysis": analysis
            }

        st.success("LULC analysis completed successfully.")

    except Exception as error:
        st.session_state.pop("lulc_results", None)
        st.error(f"Analysis failed: {error}")
        st.exception(error)

# --------------------------------------------------
# RESULTS
# --------------------------------------------------

results = st.session_state.get("lulc_results")

if results:

    analysis = results["analysis"]
    extraction = results["extraction"]
    monthly_maps = extraction["maps"]

    st.markdown(
        '<div class="section-title">📊 LULC Analysis Dashboard</div>',
        unsafe_allow_html=True
    )

    st.caption(
        f"Location: {results['latitude']:.5f}, "
        f"{results['longitude']:.5f} | "
        "Area: 1 km × 1 km | "
        "Period: October 2025 – September 2026"
    )

    # Summary metrics
    summary1, summary2, summary3, summary4 = st.columns(4)

    summary1.metric(
        "Pixels analysed",
        f"{analysis['total_pixels']:,}"
    )

    summary2.metric(
        "Changed pixels",
        f"{analysis['changed_pixels']:,}"
    )

    summary3.metric(
        "Changed area",
        f"{analysis['changed_pixels'] / 100:.2f} ha"
    )

    summary4.metric(
        "Changed percentage",
        f"{analysis['changed_percentage']:.2f}%"
    )

    st.divider()

    tab1, tab2, tab3, tab4 = st.tabs([
        "Monthly Maps",
        "Class Statistics",
        "Change Analysis",
        "Downloads"
    ])

    # --------------------------------------------------
    # TAB 1: MONTHLY MAPS
    # --------------------------------------------------

    with tab1:

        st.subheader("Monthly LULC Classification")

        selected_month = st.selectbox(
            "Select month",
            options=MONTHS,
            format_func=lambda x: MONTH_LABELS[x],
            key="map_month"
        )

        map_array = monthly_maps[selected_month]


        cmap = ListedColormap([
                LULC_COLORS[LULC_CLASS_IDS[i]]
                for i in range(1, 7) ])
        norm = BoundaryNorm(
            [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5],
            cmap.N
        )

        fig, ax = plt.subplots(figsize=(8, 7))

        ax.imshow(
            map_array,
            cmap=cmap,
            norm=norm,
            interpolation="nearest"
        )

        ax.set_title(
            f"LULC Map – {MONTH_LABELS[selected_month]}"
        )
        ax.set_axis_off()

        legend_handles = [
            plt.Line2D(
                [0], [0],
                marker="s",
                color="w",
                markerfacecolor=CLASS_COLORS[i],
                markersize=10,
                label=CLASS_NAMES[i + 1]
            )
            for i in range(6)
        ]

        ax.legend(
            handles=legend_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.03),
            ncol=3,
            frameon=False
        )

        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.caption(
            "Class predictions are derived from Sentinel-1 VV/VH "
            "radar observations using the trained V5 U-Net model."
        )

    # --------------------------------------------------
    # TAB 2: CLASS STATISTICS
    # --------------------------------------------------

    with tab2:

        st.subheader("Monthly Class Percentages")

        percentages = analysis["monthly_percentages"].copy()

        percentages["Month"] = percentages["month"].map(
            MONTH_LABELS
        )

        class_columns = [
            c for c in percentages.columns
            if c not in ["month", "Month"]
        ]

        chart_data = percentages.melt(
            id_vars=["Month"],
            value_vars=class_columns,
            var_name="Land-cover class",
            value_name="Area percentage"
        )

        fig = px.line(
                chart_data,
                x="Month",
                y="Area percentage",
                color="Land-cover class",
                markers=True,
                title="Land-cover composition over 12 months",
                color_discrete_map=LULC_COLORS,
                category_orders={
                    "Land-cover class": list(LULC_COLORS.keys())
                }
            )

        fig.update_layout(
            xaxis_title="Month",
            yaxis_title="Area (%)",
            hovermode="x unified"
        )

        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Monthly Statistics")
        st.dataframe(
            percentages.drop(columns=["Month"]),
            use_container_width=True,
            hide_index=True
        )

    # --------------------------------------------------
    # TAB 3: CHANGE ANALYSIS
    # --------------------------------------------------

    with tab3:

        st.subheader("Month-to-Month Changes")

        monthly_changes = analysis["monthly_changes"]

        fig = px.bar(
            monthly_changes,
            x="period",
            y="changed_percentage",
            title="Percentage of pixels changing class each month",
            labels={
                "period": "Period",
                "changed_percentage": "Changed pixels (%)"
            }
        )

        st.plotly_chart(fig, use_container_width=True)

        st.subheader("October 2025 – September 2026 Transition Matrix")
        
        transition_matrix = analysis[
            "first_last_transition_matrix"
        ]
        
        # Add meaningful row and column labels
        class_names = [
            LULC_CLASS_IDS[i] for i in range(1, 7)
        ]
        
        transition_df = pd.DataFrame(
            transition_matrix,
            index=class_names,
            columns=class_names
        )
        
        # Apply destination-class colours to each column
        def style_transition_columns(column):
            colour = LULC_COLORS[column.name]
            return [
                f"background-color: {colour}25"
                for _ in column
            ]
        
        styled_transition = (
            transition_df
            .style
            .format("{:,.0f}")
            .apply(style_transition_columns, axis=0)
        )
        
        st.caption(
            "Rows = October 2025 starting class | "
            "Columns = September 2026 ending class. "
            "Cell values represent pixel counts."
        )
        
        st.dataframe(
            styled_transition,
            use_container_width=True
        )


        
        # --------------------------------------------------
        # TEMPORAL STABILITY
        # --------------------------------------------------

        st.subheader("Temporal Stability")

        stability = analysis["stability"]
        total_pixels = analysis["total_pixels"]

        stable_count = stability["stable_all_months"]
        two_count = stability["two_distinct_classes"]
        multi_count = stability["three_or_more_distinct_classes"]

        stable_col, two_col, multi_col = st.columns(3)

        stable_col.metric(
            "Stable throughout",
            f"{stable_count:,}"
        )
        stable_col.caption(
            f"{stable_count / total_pixels * 100:.2f}% of selected pixels"
        )

        two_col.metric(
            "Two distinct classes",
            f"{two_count:,}"
        )
        two_col.caption(
            f"{two_count / total_pixels * 100:.2f}% of selected pixels"
        )

        multi_col.metric(
            "Three or more classes",
            f"{multi_count:,}"
        )
        multi_col.caption(
            f"{multi_count / total_pixels * 100:.2f}% of selected pixels"
        )

        # --------------------------------------------------
        # PIXEL SWITCH DISTRIBUTION
        # --------------------------------------------------

        st.subheader("Pixel Switch Distribution")

        switch_data = analysis["switch_distribution"]

        fig = px.bar(
            switch_data,
            x="switches",
            y="percentage",
            title="Temporal Class Changes per Pixel",
            labels={
                "switches": "Number of class switches",
                "percentage": "Pixels (%)"
            },
            hover_data={
                "switches": True,
                "percentage": ":.2f"
            }
        )

        fig.update_traces(
            hovertemplate=(
                "Class switches: %{x}<br>"
                "Pixels: %{y:.2f}%"
                "<extra></extra>"
            )
        )

        fig.update_layout(
            xaxis=dict(
                title="Number of class switches",
                dtick=1
            ),
            yaxis=dict(
                title="Pixels (%)"
            ),
            bargap=0.25
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        # --------------------------------------------------
        # TRANSITION DETAILS
        # --------------------------------------------------

        st.subheader("Transition Details")

        transition_details = analysis["transition_details"].rename(
            columns={
                "from_class": "Starting class",
                "to_class": "Ending class",
                "pixels": "Pixel count",
                "area_hectares": "Area (hectares)"
            }
        )

        styled_details = transition_details.style.format({
            "Pixel count": "{:,.0f}",
            "Area (hectares)": "{:.2f}"
        })

        st.dataframe(
            styled_details,
            use_container_width=True,
            hide_index=True
        )

        st.warning(
            "These are changes in model predictions. They should not "
            "be interpreted as independently verified physical "
            "land-cover conversions."
        )

    # --------------------------------------------------
    # TAB 4: DOWNLOADS
    # --------------------------------------------------

    with tab4:

        st.subheader("Download Monthly LULC Maps")

        for month in MONTHS:

            output_path = extraction["output_paths"][month]

            if Path(output_path).exists():

                with open(output_path, "rb") as file:
                    st.download_button(
                        label=f"Download {MONTH_LABELS[month]} GeoTIFF",
                        data=file.read(),
                        file_name=f"lulc_{month}_selected_area.tif",
                        mime="image/tiff",
                        key=f"download_{month}"
                    )

        st.divider()

        st.subheader("Download Statistics")

        csv_files = {
            "Monthly class percentages": (
                analysis["monthly_percentages"],
                "monthly_class_percentages.csv"
            ),
            "Monthly changes": (
                analysis["monthly_changes"],
                "monthly_changes.csv"
            ),
            "Transition details": (
                analysis["transition_details"],
                "transition_details.csv"
            ),
            "Transition matrix": (
                analysis["first_last_transition_matrix"],
                "transition_matrix.csv"
            )
        }

        for label, (dataframe, filename) in csv_files.items():

            st.download_button(
                label=f"Download {label} CSV",
                data=dataframe.to_csv().encode("utf-8"),
                file_name=filename,
                mime="text/csv",
                key=f"csv_{filename}"
            )

else:

    st.info(
        "Enter a location in the sidebar and click "
        "'Generate LULC Analysis' to start."
    )

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.markdown(
    '<div class="status-note">'
    'Prototype: V5 model | Sentinel-1 VV/VH | '
    'Current coverage: Stuttgart 40 km × 40 km'
    '</div>',
    unsafe_allow_html=True
)
