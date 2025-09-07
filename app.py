# pip install streamlit pandas openpyxl geopandas folium streamlit-folium shapely plotly

import streamlit as st
import pandas as pd
import geopandas as gpd
import folium
from streamlit_folium import st_folium
from folium.features import GeoJsonTooltip
from io import BytesIO
from pathlib import Path
import plotly.graph_objects as go
from pathlib import Path

# ========== Paths ==========
ROOT = Path(__file__).resolve().parent
BASE_DIR = ROOT / "data"

FILEMAP = {
    "pre":  BASE_DIR / "pre_area weight.xlsx",
    "tmax": BASE_DIR / "tmax_area weight.xlsx",
    "tmin": BASE_DIR / "tmin_area weight.xlsx",
    "sm":   BASE_DIR / "sm_area weight.xlsx",
}
SHEET_NAME = "annual mean"   # 你的年表 sheet 名

# 流域边界（用相对路径）
CATCHMENT_SHP = BASE_DIR / "catchments" / "hrs_467_catchms.shp"
st.set_page_config(page_title="HRS | Annual Means (pre / tmax / tmin / sm)", layout="wide")
st.title("HRS Basin Annual Means (1950–2024): Temporal & Spatial Comparison")

# -------------------- Loaders --------------------
@st.cache_data(show_spinner=False)
def load_wide_excel(path: str | Path, sheet_name: str = SHEET_NAME):
    """Read wide-format Excel (StationID[, State], years...) and normalize year columns to INT."""
    df = pd.read_excel(path, sheet_name=sheet_name)

    # Normalize year-like columns to int (e.g., '1961' -> 1961)
    year_like = [c for c in df.columns if str(c).isdigit()]
    rename_map = {c: int(c) for c in year_like}
    df = df.rename(columns=rename_map)

    year_cols = sorted([c for c in df.columns if isinstance(c, int)])
    base_cols = [c for c in ["StationID", "State"] if c in df.columns]
    return df[base_cols + year_cols], year_cols

@st.cache_data(show_spinner=False)
def load_all_variables(filemap: dict[str, Path]):
    data = {}
    years_sets = []
    for var, p in filemap.items():
        df, years = load_wide_excel(p, SHEET_NAME)
        data[var] = {"df": df, "years": years}
        years_sets.append(set(years))
    common_years = sorted(list(set.intersection(*years_sets))) if years_sets else []
    return data, common_years

@st.cache_data(show_spinner=False)
def load_gdf(path):
    gdf = gpd.read_file(path)
    if gdf.crs is None:
        gdf.set_crs(epsg=4326, inplace=True)
    else:
        gdf = gdf.to_crs(epsg=4326)
    if "StationID" not in gdf.columns:
        cand = [c for c in gdf.columns if "station" in c.lower() or c.lower().endswith("id")]
        if cand: gdf = gdf.rename(columns={cand[0]: "StationID"})
        else: raise ValueError("No StationID field in shapefile.")
    if "State" not in gdf.columns:
        sc = [c for c in gdf.columns if "state" in c.lower()]
        if sc: gdf = gdf.rename(columns={sc[0]: "State"})
        else: gdf["State"] = None
    return gdf

DATA, COMMON_YEARS = load_all_variables(FILEMAP)
gdf = load_gdf(CATCHMENT_SHP)

# Intersect StationIDs across all variables + shapefile
id_sets = [set(DATA[var]["df"]["StationID"]) for var in DATA]
id_sets.append(set(gdf["StationID"]))
common_ids = sorted(list(set.intersection(*id_sets))) if id_sets else []

# Keep only common StationIDs & years
for var in DATA:
    df = DATA[var]["df"]
    years = DATA[var]["years"]
    keep_cols = [c for c in ["StationID", "State"] if c in df.columns] + [y for y in years if y in COMMON_YEARS]
    DATA[var]["df"] = df[df["StationID"].isin(common_ids)][keep_cols].copy()
gdf = gdf[gdf["StationID"].isin(common_ids)].copy()

# -------------------- Sidebar --------------------
with st.sidebar:
    st.header("Filters")

    # Time series modes
    ts_mode = st.radio(
        "Time series mode",
        ["Multi-stations · Single variable", "Single station · Multi-variables"],
        index=0,
    )

    var_options = ["pre", "tmax", "tmin", "sm"]

    if ts_mode == "Multi-stations · Single variable":
        ts_var = st.selectbox("Variable (for time series)", options=var_options, index=0)
        stations_for_ts = st.multiselect(
            "Select StationID(s) to compare",
            options=common_ids,
            default=common_ids[:3] if len(common_ids) >= 3 else common_ids,
        )
    else:
        station_sel = st.selectbox("StationID (for time series)", options=common_ids)
        vars_for_ts = st.multiselect(
            "Select variables to compare",
            options=var_options,
            default=["pre", "sm"]
        )
        use_dual_axis = st.checkbox("Use dual y-axes when selecting exactly two variables (except TMIN & TMAX pair)", value=True)
        right_axis_var = None
        if use_dual_axis and len(vars_for_ts) == 2:
            right_axis_var = st.selectbox(
                "Variable on RIGHT y-axis (for non TMIN–TMAX pairs)",
                options=[v.upper() for v in vars_for_ts],
                index=1
            )

    # Map controls
    var_for_map = st.selectbox("Variable for map", options=var_options, index=0)
    year_for_map = st.slider(
        "Year (for map)",
        min_value=min(COMMON_YEARS), max_value=max(COMMON_YEARS),
        value=min(COMMON_YEARS), step=1
    )
    color_mode = st.radio("Map color classification", ["Quantile-based (adaptive)", "Fixed thresholds"], index=0)
    bins_text = st.text_input(
        "Fixed thresholds (comma separated, ascending)",
        "0,100,250,500,750,1000,1250,1500"
    ) if color_mode == "Fixed thresholds" else None

# -------------------- Tabs --------------------
tab1, tab2 = st.tabs([
    "📈 Time Series",
    f"🗺 Spatial Distribution ({year_for_map}, {var_for_map})"
])

# ===== Tab 1: Time series =====
with tab1:
    if ts_mode == "Multi-stations · Single variable":
        st.subheader(f"Annual means — {ts_var.upper()} (compare multiple basins)")
        if not stations_for_ts:
            st.info("Pick at least one StationID in the sidebar.")
        else:
            df = DATA[ts_var]["df"]
            sub = df[df["StationID"].isin(stations_for_ts)].copy()
            long_rows = []
            for _, r in sub.iterrows():
                sid = r["StationID"]
                for c in sub.columns:
                    if isinstance(c, int):  # year col
                        long_rows.append({"Year": c, "Series": str(sid), "Value": float(r[c])})
            if long_rows:
                ts_df = pd.DataFrame(long_rows).sort_values(["Series", "Year"])
                plot_df = ts_df.pivot(index="Year", columns="Series", values="Value")
                st.line_chart(plot_df, height=420)
                st.caption(f"Comparing multiple basins for **{ts_var.upper()}** (annual mean).")
                st.download_button(
                    "Download time series (CSV)",
                    ts_df.to_csv(index=False).encode("utf-8-sig"),
                    file_name=f"{ts_var}_annual_means_multi_stations.csv",
                    mime="text/csv"
                )
            else:
                st.warning("No data for the selected basins.")
    else:
        st.subheader(f"Annual means — Station {station_sel} (compare variables)")
        if not vars_for_ts:
            st.info("Pick at least one variable.")
        else:
            rows = []
            for var in vars_for_ts:
                df = DATA[var]["df"]
                row = df[df["StationID"] == station_sel]
                if row.empty:
                    continue
                for c in row.columns:
                    if isinstance(c, int):
                        rows.append({"Year": c, "Variable": var.upper(), "Value": float(row.iloc[0][c])})
            if not rows:
                st.warning("No data for this station/variables.")
            else:
                ts_df = pd.DataFrame(rows).sort_values(["Variable", "Year"])
                selected_vars = sorted(ts_df["Variable"].unique())

                # --- Rule: if exactly TMIN & TMAX -> single axis; else dual axis (if exactly two vars & enable) ---
                is_t_pair = set(selected_vars) == {"TMIN", "TMAX"}

                if len(selected_vars) == 2 and not is_t_pair and use_dual_axis:
                    # dual y-axes (e.g., PRE vs SM)
                    vars_sorted = selected_vars
                    right = right_axis_var if right_axis_var else vars_sorted[1]
                    left = [v for v in vars_sorted if v != right][0]

                    left_df = ts_df[ts_df["Variable"] == left]
                    right_df = ts_df[ts_df["Variable"] == right]

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=left_df["Year"], y=left_df["Value"],
                        mode="lines", name=left, yaxis="y1"
                    ))
                    fig.add_trace(go.Scatter(
                        x=right_df["Year"], y=right_df["Value"],
                        mode="lines", name=right, yaxis="y2"
                    ))
                    fig.update_layout(
                        height=460,
                        xaxis=dict(title="Year", gridcolor="#3a3a3a"),
                        yaxis=dict(title=f"{left} (left axis)", showgrid=True, gridcolor="#3a3a3a"),
                        yaxis2=dict(
                            title=f"{right} (right axis)",
                            overlaying="y", side="right",
                            showgrid=False
                        ),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                        # transparent backgrounds => follow Streamlit theme (dark)
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    st.caption("Dual y-axes for two variables with different scales (right axis grid hidden).")

                else:
                    # single-axis (includes TMIN & TMAX pair, or >=3 variables)
                    plot_df = ts_df.pivot(index="Year", columns="Variable", values="Value")
                    # Use Plotly for better dark-theme look and a single grid set
                    fig = go.Figure()
                    for col in plot_df.columns:
                        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df[col], mode="lines", name=col))
                    fig.update_layout(
                        height=460,
                        xaxis=dict(title="Year", gridcolor="#3a3a3a"),
                        yaxis=dict(title="Annual mean", gridcolor="#3a3a3a"),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    st.caption("Single y-axis. TMIN & TMAX pair is plotted on the same axis by design.")

                st.download_button(
                    "Download time series (CSV)",
                    ts_df.sort_values(["Variable", "Year"]).to_csv(index=False).encode("utf-8-sig"),
                    file_name=f"{station_sel}_annual_means_multi_vars.csv",
                    mime="text/csv"
                )

# ===== Tab 2: Map =====
with tab2:
    st.subheader(f"Spatial distribution: {var_for_map.upper()} in {year_for_map}")
    df_map_var = DATA[var_for_map]["df"]

    if year_for_map not in df_map_var.columns:
        available = [c for c in df_map_var.columns if isinstance(c, int)]
        st.warning(f"No column for year {year_for_map} in {var_for_map}. "
                   f"Available years: {available[:15]}{' ...' if len(available)>15 else ''}")
    else:
        map_df = gdf.merge(
            df_map_var[["StationID", year_for_map]],
            on="StationID", how="left"
        ).rename(columns={year_for_map: "AnnualMean"})

        minx, miny, maxx, maxy = map_df.total_bounds
        center = [(miny + maxy)/2.0, (minx + maxx)/2.0]
        m = folium.Map(location=center, zoom_start=4, tiles="cartodbpositron")

        if color_mode == "Fixed thresholds":
            try:
                bins = [float(x.strip()) for x in bins_text.split(",") if x.strip() != ""]
                bins = sorted(bins)
            except Exception:
                st.warning("Failed to parse thresholds. Reverting to quantile mode.")
                bins = None
        else:
            bins = None

        folium.Choropleth(
            geo_data=map_df,
            data=map_df,
            columns=["StationID", "AnnualMean"],
            key_on="feature.properties.StationID",
            fill_color="YlGnBu",
            fill_opacity=0.85,
            line_opacity=0.2,
            nan_fill_color="#f2f2f2",
            legend_name=f"{var_for_map.upper()} annual mean in {year_for_map}",
            name="AnnualMean"
        ).add_to(m)

        folium.GeoJson(
            map_df,
            tooltip=GeoJsonTooltip(
                fields=["StationID", "State", "AnnualMean"],
                aliases=["StationID", "State", f"AnnualMean {year_for_map}"],
                localize=True
            ),
            style_function=lambda x: {"color": "black", "weight": 0.3, "fillOpacity": 0},
            highlight_function=lambda x: {"weight": 2, "color": "black"},
            name="Boundaries"
        ).add_to(m)

        # no LayerControl -> no layer switcher box
        st_folium(m, use_container_width=True, height=620)

        html_bytes = BytesIO(m.get_root().render().encode("utf-8"))
        st.download_button(
            "Download current map (HTML)",
            data=html_bytes,
            file_name=f"{var_for_map}_{year_for_map}_annual_mean_map.html",
            mime="text/html"
        )

