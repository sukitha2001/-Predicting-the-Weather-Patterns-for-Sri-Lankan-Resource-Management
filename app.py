import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import joblib
import os
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Set page configuration
st.set_page_config(
    page_title="Sri Lanka Climate Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- MODERN CSS + JS REDESIGN ---
st.markdown(
    """
    <!-- Google Fonts: Inter -->
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap" rel="stylesheet">

    <style>
    /* =====================================================
       GLOBAL RESETS & TYPOGRAPHY
    ===================================================== */
    html, body, [class*="css"], .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        background-color: #06070D !important;
        color: #E8EAED !important;
    }

    /* Main content area */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 3rem !important;
        max-width: 1400px !important;
    }

    /* =====================================================
       HEADINGS — BOLD & LARGE
    ===================================================== */
    h1, .stTitle, [data-testid="stHeadingWithActionElements"] h1 {
        font-size: 2.8rem !important;
        font-weight: 900 !important;
        letter-spacing: -0.03em !important;
        background: linear-gradient(135deg, #FFFFFF 0%, #A8B8FF 100%) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        background-clip: text !important;
        line-height: 1.15 !important;
        margin-bottom: 0.4rem !important;
    }

    h2 {
        font-size: 2rem !important;
        font-weight: 800 !important;
        letter-spacing: -0.02em !important;
        color: #FFFFFF !important;
    }

    h3, .stMarkdown h3 {
        font-size: 1.35rem !important;
        font-weight: 700 !important;
        letter-spacing: -0.01em !important;
        color: #C8D0FF !important;
        margin-top: 1.5rem !important;
        margin-bottom: 0.75rem !important;
    }

    h4 {
        font-size: 1.1rem !important;
        font-weight: 700 !important;
        color: #A8B8FF !important;
    }

    p, .stMarkdown p, label, .stSelectbox label, .stMultiSelect label {
        font-size: 1rem !important;
        font-weight: 500 !important;
        line-height: 1.7 !important;
        color: #B0BCC8 !important;
    }

    /* =====================================================
       SIDEBAR — DARK GLASS PANEL
    ===================================================== */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0D0F1A 0%, #111420 100%) !important;
        border-right: 1px solid rgba(168, 184, 255, 0.12) !important;
        box-shadow: 4px 0 30px rgba(0, 0, 0, 0.6) !important;
    }

    [data-testid="stSidebar"] h2 {
        font-size: 1.4rem !important;
        font-weight: 800 !important;
        color: #FFFFFF !important;
        -webkit-text-fill-color: #FFFFFF !important;
        background: none !important;
    }

    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] .stMarkdown p {
        font-size: 0.88rem !important;
        color: #8090A8 !important;
    }

    [data-testid="stSidebar"] hr {
        border-color: rgba(168, 184, 255, 0.15) !important;
        margin: 1rem 0 !important;
    }

    /* Sidebar section label */
    [data-testid="stSidebar"] h3 {
        font-size: 0.75rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.1em !important;
        text-transform: uppercase !important;
        color: #5060A0 !important;
        margin-bottom: 0.5rem !important;
    }

    /* Lock sidebar — hide collapse buttons */
    [data-testid="stSidebarCollapseButton"],
    [data-testid="collapsedControl"] {
        display: none !important;
    }

    /* =====================================================
       SELECTBOX & MULTISELECT — STYLED INPUTS
    ===================================================== */
    [data-testid="stSelectbox"] > div > div,
    [data-testid="stMultiSelect"] > div > div {
        background: rgba(255,255,255,0.04) !important;
        border: 1.5px solid rgba(168, 184, 255, 0.2) !important;
        border-radius: 10px !important;
        color: #E8EAED !important;
        font-size: 0.95rem !important;
        font-weight: 600 !important;
        transition: border-color 0.2s ease !important;
    }

    [data-testid="stSelectbox"] > div > div:hover,
    [data-testid="stMultiSelect"] > div > div:hover {
        border-color: rgba(168, 184, 255, 0.5) !important;
    }

    /* =====================================================
       KPI METRIC CARDS — GLASSMORPHISM
    ===================================================== */
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, rgba(255,255,255,0.06) 0%, rgba(168,184,255,0.04) 100%) !important;
        border: 1px solid rgba(168, 184, 255, 0.18) !important;
        border-radius: 16px !important;
        padding: 22px 20px !important;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4), inset 0 1px 0 rgba(255,255,255,0.08) !important;
        backdrop-filter: blur(12px) !important;
        transition: transform 0.2s ease, box-shadow 0.2s ease !important;
    }

    div[data-testid="metric-container"]:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 16px 48px rgba(0, 0, 0, 0.5), 0 0 0 1px rgba(168,184,255,0.3) !important;
    }

    div[data-testid="metric-container"] [data-testid="stMetricLabel"] {
        font-size: 0.78rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.08em !important;
        text-transform: uppercase !important;
        color: #6070A0 !important;
    }

    div[data-testid="metric-container"] [data-testid="stMetricValue"] {
        font-size: 2rem !important;
        font-weight: 900 !important;
        color: #FFFFFF !important;
        letter-spacing: -0.03em !important;
        line-height: 1.2 !important;
    }

    /* =====================================================
       EXPANDER — CLEAN ACCORDION
    ===================================================== */
    [data-testid="stExpander"] {
        background: rgba(255,255,255,0.03) !important;
        border: 1px solid rgba(168, 184, 255, 0.12) !important;
        border-radius: 14px !important;
        overflow: hidden !important;
    }

    [data-testid="stExpander"] summary {
        font-size: 1rem !important;
        font-weight: 700 !important;
        color: #C8D0FF !important;
        padding: 1rem 1.25rem !important;
    }

    /* =====================================================
       BUTTONS
    ===================================================== */
    .stButton > button,
    [data-testid="stFormSubmitButton"] > button {
        background: linear-gradient(135deg, #4A5AFF 0%, #7B8BFF 100%) !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 10px !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.95rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.02em !important;
        padding: 0.6rem 1.4rem !important;
        box-shadow: 0 4px 20px rgba(74, 90, 255, 0.35) !important;
        transition: all 0.2s ease !important;
        cursor: pointer !important;
    }

    .stButton > button:hover,
    [data-testid="stFormSubmitButton"] > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 30px rgba(74, 90, 255, 0.55) !important;
        background: linear-gradient(135deg, #5A6AFF 0%, #8B9BFF 100%) !important;
    }

    /* Download button */
    [data-testid="stDownloadButton"] > button {
        background: rgba(255,255,255,0.05) !important;
        color: #A8B8FF !important;
        border: 1.5px solid rgba(168, 184, 255, 0.3) !important;
        border-radius: 10px !important;
        font-weight: 700 !important;
        font-size: 0.9rem !important;
        transition: all 0.2s ease !important;
    }

    [data-testid="stDownloadButton"] > button:hover {
        background: rgba(168, 184, 255, 0.1) !important;
        border-color: rgba(168, 184, 255, 0.6) !important;
    }

    /* =====================================================
       NUMBER INPUT
    ===================================================== */
    [data-testid="stNumberInput"] input {
        background: rgba(255,255,255,0.05) !important;
        border: 1.5px solid rgba(168,184,255,0.2) !important;
        border-radius: 10px !important;
        color: #FFFFFF !important;
        font-size: 1rem !important;
        font-weight: 600 !important;
    }

    /* =====================================================
       DATAFRAME / TABLE
    ===================================================== */
    [data-testid="stDataFrame"] {
        border: 1px solid rgba(168, 184, 255, 0.12) !important;
        border-radius: 12px !important;
        overflow: hidden !important;
    }

    /* =====================================================
       INFO / ALERT BOXES
    ===================================================== */
    [data-testid="stInfo"] {
        background: rgba(74, 90, 255, 0.08) !important;
        border: 1px solid rgba(74, 90, 255, 0.25) !important;
        border-radius: 12px !important;
        font-size: 0.9rem !important;
    }

    [data-testid="stWarning"] {
        background: rgba(255, 180, 50, 0.08) !important;
        border: 1px solid rgba(255, 180, 50, 0.25) !important;
        border-radius: 12px !important;
    }

    [data-testid="stSuccess"] {
        background: rgba(0, 230, 118, 0.08) !important;
        border: 1px solid rgba(0, 230, 118, 0.25) !important;
        border-radius: 12px !important;
        font-size: 1rem !important;
        font-weight: 600 !important;
    }

    /* =====================================================
       HORIZONTAL DIVIDER
    ===================================================== */
    hr {
        border: none !important;
        border-top: 1px solid rgba(168, 184, 255, 0.1) !important;
        margin: 2rem 0 !important;
    }

    /* =====================================================
       HEADER — HIDE TOOLBAR & FOOTER
    ===================================================== */
    [data-testid="stToolbar"] { visibility: hidden !important; }
    footer { visibility: hidden !important; }
    [data-testid="stHeader"] { background: transparent !important; }

    /* =====================================================
       PLOTLY CHART CONTAINER
    ===================================================== */
    [data-testid="stPlotlyChart"] {
        border-radius: 16px !important;
        overflow: hidden !important;
        border: 1px solid rgba(168, 184, 255, 0.1) !important;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.35) !important;
    }

    /* =====================================================
       INSIGHT CALLOUT BOXES
       Target the blockquote-style insight paragraphs
    ===================================================== */
    .stMarkdown p:has(> strong:first-child) {
        background: rgba(168, 184, 255, 0.05) !important;
        border-left: 3px solid #4A5AFF !important;
        border-radius: 0 8px 8px 0 !important;
        padding: 0.75rem 1rem !important;
        margin: 0.5rem 0 !important;
        font-size: 0.95rem !important;
    }

    /* =====================================================
       FORM CONTAINER
    ===================================================== */
    [data-testid="stForm"] {
        background: rgba(255,255,255,0.03) !important;
        border: 1px solid rgba(168,184,255,0.12) !important;
        border-radius: 16px !important;
        padding: 1.5rem !important;
    }

    /* =====================================================
       COLUMNS GAP
    ===================================================== */
    [data-testid="column"] {
        gap: 1rem !important;
    }

    /* =====================================================
       SCROLLBAR STYLING
    ===================================================== */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb {
        background: rgba(168, 184, 255, 0.25);
        border-radius: 3px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(168, 184, 255, 0.5);
    }

    </style>

    <script>
    // =====================================================
    // JS: Animate metric cards on load + scroll-in effect
    // =====================================================
    (function() {
        function animateOnLoad() {
            // Fade-in metrics with stagger
            const style = document.createElement('style');
            style.textContent = `
                @keyframes fadeInUp {
                    from { opacity: 0; transform: translateY(20px); }
                    to   { opacity: 1; transform: translateY(0); }
                }
                @keyframes slideInLeft {
                    from { opacity: 0; transform: translateX(-16px); }
                    to   { opacity: 1; transform: translateX(0); }
                }
                @keyframes glowPulse {
                    0%, 100% { box-shadow: 0 8px 32px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.08); }
                    50% { box-shadow: 0 8px 32px rgba(0,0,0,0.4), 0 0 20px rgba(74,90,255,0.15), inset 0 1px 0 rgba(255,255,255,0.08); }
                }

                div[data-testid="metric-container"] {
                    animation: fadeInUp 0.5s ease forwards, glowPulse 4s ease-in-out infinite !important;
                }

                h1, .stTitle { animation: fadeInUp 0.4s ease forwards !important; }

                [data-testid="stSidebar"] {
                    animation: slideInLeft 0.5s ease forwards !important;
                }

                [data-testid="stPlotlyChart"] {
                    animation: fadeInUp 0.6s ease forwards !important;
                }
            `;
            document.head.appendChild(style);

            // Stagger metric cards
            const observer = new MutationObserver(() => {
                const metrics = document.querySelectorAll('div[data-testid="metric-container"]');
                metrics.forEach((el, i) => {
                    el.style.animationDelay = (i * 0.08) + 's';
                });
            });
            observer.observe(document.body, { childList: true, subtree: true });
        }

        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', animateOnLoad);
        } else {
            animateOnLoad();
        }
    })();
    </script>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# Data Loading & Caching
# ---------------------------------------------------------
@st.cache_data
def load_data():
    file_path = "Data/Cleaned_SriLanka_Weather_Dataset.csv"
    try:
        df = pd.read_csv(file_path, parse_dates=["time"])
        return df
    except FileNotFoundError:
        st.error(f"Dataset not found at {file_path}. Please check the Data folder.")
        return pd.DataFrame()


WEATHER = load_data()

# ---------------------------------------------------------
# Sidebar Navigation & Global Filters
# ---------------------------------------------------------
with st.sidebar:
    # Header
    st.markdown(
        """
        <div style='text-align: center; padding-bottom: 20px;'>
            <h2 style='color: #FAFAFA; margin-bottom: 0px;'>ST3011 Project</h2>
            <p style='color: #00E676; font-size: 14px; font-weight: 600; letter-spacing: 1px;'>CLIMATE INTELLIGENCE</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Navigation
    st.markdown("### 📍 Menu")
    page = st.selectbox(
        "Select Page",
        [
            "EDA Dashboard",
            "Time Series Analysis",
            "K-Means Clustering",
            "SARIMA Forecasts",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")

    # Filters
    st.markdown("### 🔍 Global Filters")
    if not WEATHER.empty:
        all_cities = sorted(WEATHER["city"].dropna().unique())
        selected_cities = st.multiselect(
            "Compare Cities (Leave blank for all):", options=all_cities, default=[]
        )

        if selected_cities:
            filtered_weather = WEATHER[WEATHER["city"].isin(selected_cities)]
        else:
            filtered_weather = WEATHER
    else:
        filtered_weather = pd.DataFrame()

    st.markdown("---")

    # About Section
    st.markdown("### ℹ️ About this Project")
    st.info("""
    **Developer:** Sukitha Rathnayake  
    **Module:** ST3011  
    
    This interactive dashboard is designed to perform advanced time series analysis on Sri Lankan weather data. 
    
    By mapping historical climate profiles, seasonal decomposition, and SARIMA forecasting models, this tool aims to provide actionable intelligence for understanding climate effects on **renewable energy generation**.
    """)

    # Footer
    st.markdown(
        """
        <div style='text-align: center; color: #5c6370; font-size: 12px; margin-top: 20px;'>
            Built with Streamlit & Plotly
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------
# Page 1: Exploratory Data Analysis (EDA)
# ---------------------------------------------------------
if page == "EDA Dashboard":
    st.title("Exploratory Data Analysis")
    st.markdown(
        "A comprehensive overview of historical patterns, distributions, and correlations in the Sri Lankan climate data."
    )

    if filtered_weather.empty:
        st.warning("No data available.")
        st.stop()

    # Raw Data Viewer
    with st.expander("🔎 View Cleaned Dataset (Tabular Format)"):
        st.markdown(
            "Explore the raw records driving this dashboard. The table below automatically updates based on your active sidebar filters."
        )
        st.dataframe(filtered_weather, use_container_width=True, height=250)

        csv_data = filtered_weather.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Filtered Data as CSV",
            data=csv_data,
            file_name="filtered_climate_data.csv",
            mime="text/csv",
        )

    st.markdown("---")

    # SECTION 1: KPI METRIC CARDS
    st.markdown("### Key Climate Metrics")
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric(
        label="Average Temp",
        value=f"{filtered_weather['temperature_2m_mean'].mean():.1f} °C",
    )
    kpi2.metric(
        label="Total Precipitation",
        value=f"{filtered_weather['precipitation_sum'].sum():,.0f} mm",
    )
    kpi3.metric(
        label="Max Wind Speed",
        value=f"{filtered_weather['windspeed_10m_max'].max():.1f} km/h",
    )
    kpi4.metric(
        label="Avg Evapotranspiration",
        value=f"{filtered_weather['et0_fao_evapotranspiration'].mean():.2f} mm",
    )

    st.markdown("---")

    # SECTION 2: WIDE PLOT (PRECIPITATION)
    st.markdown("### 🌧️ Precipitation Trends")
    if "Wet Zone" in filtered_weather["zone"].values:
        zonal_daily = (
            filtered_weather.groupby(["time", "zone"])[["precipitation_sum"]]
            .mean(numeric_only=True)
            .unstack()
        )
        zonal_daily.columns = [f"{col[0]}_{col[1]}" for col in zonal_daily.columns]
        annual_rain = zonal_daily.resample("YE").sum().reset_index()
        annual_rain["year"] = annual_rain["time"].dt.year

        if "precipitation_sum_Wet Zone" in annual_rain.columns:
            baseline = annual_rain["precipitation_sum_Wet Zone"].mean()

            fig = px.bar(
                annual_rain,
                x="year",
                y="precipitation_sum_Wet Zone",
                title="Wet Zone Annual Rainfall",
                labels={
                    "precipitation_sum_Wet Zone": "Total Rainfall (mm)",
                    "year": "Year",
                },
                color_discrete_sequence=["#00D4B2"],
                template="plotly_dark",
            )
            fig.add_hline(
                y=baseline,
                line_dash="dash",
                line_color="#FF4B4B",
                annotation_text=f"Baseline: {baseline:.0f}mm",
            )
            fig.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(dtick=1),
                height=450,
            )
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("""
            💡 **Key Insights:**
            * **Deficit Identification:** Clearly highlights specific years where total rainfall fell significantly below the historical baseline (e.g., potential drought years).
            * **Cyclical Patterns:** Visualizes the broader inter-annual variability of the Wet Zone's monsoon performance over the recorded decade.
            * **Resource Planning:** Crucial for understanding long-term water availability impacts on agriculture and hydropower reservoir capacities.
            """)
    else:
        st.info("Select a city in the Wet Zone to view precipitation trends.")

    st.markdown("---")

    # SECTION 3: SPLIT COLUMNS (TEMP & WIND)
    st.markdown("### 🌡️ Temperature & Wind Patterns")
    col_t, col_w = st.columns(2)

    with col_t:
        fig2 = px.histogram(
            filtered_weather,
            x="temperature_2m_mean",
            title="Distribution of Mean Temperatures",
            nbins=30,
            marginal="box",
            color_discrete_sequence=["#FF7F50"],
            labels={"temperature_2m_mean": "Temperature (°C)"},
            template="plotly_dark",
        )
        fig2.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", height=450
        )
        st.plotly_chart(fig2, use_container_width=True)

        st.markdown("""
        💡 **Key Insights:**
        * **Central Tendency:** Displays the most frequent temperature ranges experienced across the selected regions.
        * **Extreme Events:** The tails of the distribution help identify the frequency of extreme heat or unusual cold waves.
        * **Climate Stability:** A tight, normally distributed curve indicates highly stable seasonal temperatures.
        """)

    with col_w:
        counts, bins = np.histogram(
            filtered_weather["winddirection_10m_dominant"].dropna(),
            bins=16,
            range=(0, 360),
        )
        bin_centers = bins[:-1] + np.diff(bins) / 2

        fig_polar = px.bar_polar(
            r=counts,
            theta=bin_centers,
            title="Dominant Wind Direction",
            color_discrete_sequence=["#00E676"],
            start_angle=90,
            direction="clockwise",
            template="plotly_dark",
        )
        fig_polar.update_layout(
            polar=dict(
                bgcolor="rgba(0,0,0,0)", angularaxis=dict(tickfont=dict(size=10))
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=50, b=20),
            height=450,
        )
        st.plotly_chart(fig_polar, use_container_width=True)

        st.markdown("""
        💡 **Key Insights:**
        * **Monsoon Dominance:** Illustrates the prevailing wind corridors, typically reflecting Sri Lanka's Southwest and Northeast monsoon systems.
        * **Energy Optimization:** Highly valuable for determining the optimal geographic orientation for wind turbine placements.
        * **Directional Frequency:** Shows exactly which angular degrees produce the most frequent wind gusts.
        """)

    st.markdown("---")

    # SECTION 4: WIDE PLOT (CORRELATIONS)
    st.markdown("### 📊 Variable Correlations")
    numeric_cols = filtered_weather.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        corr = filtered_weather[numeric_cols].corr()
        fig3 = px.imshow(
            corr,
            text_auto=".2f",
            aspect="auto",
            color_continuous_scale="Viridis",
            template="plotly_dark",
        )
        fig3.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", height=600
        )
        st.plotly_chart(fig3, use_container_width=True)

        st.markdown("""
        💡 **Key Insights:**
        * **Feature Relationships:** Identifies strong positive and negative linear relationships between different meteorological factors.
        * **Multicollinearity Detection:** Helps in predictive modeling by spotting highly correlated independent variables.
        * **Behavioral Validation:** Confirms expected physical climate behaviors (e.g., inverse relationship between precipitation and maximum temperatures).
        """)

# ---------------------------------------------------------
# Page 2: Time Series Models & Seasonal Decomposition
# ---------------------------------------------------------
elif page == "Time Series Analysis":
    st.title("Time Series Analysis")

    if filtered_weather.empty:
        st.stop()

    variable_choice = st.selectbox(
        "Analyze Variable:", ["Evapotranspiration", "Wind Speed"]
    )

    st.subheader(f"Smoothed 30-Day Trend: {variable_choice}")

    target_col = (
        "et0_fao_evapotranspiration"
        if variable_choice == "Evapotranspiration"
        else "windspeed_10m_max"
    )
    zone_type = "zone" if variable_choice == "Evapotranspiration" else "wind_zone"

    agg_data = (
        filtered_weather.groupby(["time", zone_type])[target_col].mean().reset_index()
    )
    agg_data["smoothed"] = agg_data.groupby(zone_type)[target_col].transform(
        lambda x: x.rolling(window=30, min_periods=1).mean()
    )

    fig_ts = px.line(
        agg_data,
        x="time",
        y="smoothed",
        color=zone_type,
        labels={"smoothed": f"Smoothed {variable_choice}", "time": "Date"},
        template="plotly_dark",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig_ts.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_ts, use_container_width=True)

    st.markdown("""
    💡 **Key Insights:**
    * **Noise Reduction:** The 30-day rolling mean effectively filters out daily meteorological "noise" to reveal true underlying trends.
    * **Zonal Discrepancies:** Clearly illustrates how the selected variable behaves differently across distinct climatic or wind zones in Sri Lanka.
    * **Macro-Volatility:** Helps in spotting broader periods of instability or sustained highs/lows over several months.
    """)

    st.markdown("---")

    st.subheader("Seasonal Decomposition & Autocorrelation")
    monthly_data = (
        filtered_weather.groupby([pd.Grouper(key="time", freq="ME"), zone_type])[
            target_col
        ]
        .mean()
        .unstack()
    )
    zone_choice = st.selectbox("Select Zone to Analyze:", monthly_data.columns.dropna())

    if pd.notna(zone_choice) and not monthly_data[zone_choice].dropna().empty:
        # 1. Seasonal Decomposition
        res = seasonal_decompose(
            monthly_data[zone_choice].dropna(), model="additive", period=12
        )
        from plotly.subplots import make_subplots

        fig_dec = make_subplots(
            rows=4,
            cols=1,
            shared_xaxes=True,
            subplot_titles=("Observed", "Trend", "Seasonal", "Residual"),
        )

        fig_dec.add_trace(
            go.Scatter(
                x=res.observed.index,
                y=res.observed,
                name="Observed",
                line=dict(color="#00D4B2"),
            ),
            row=1,
            col=1,
        )
        fig_dec.add_trace(
            go.Scatter(
                x=res.trend.index, y=res.trend, name="Trend", line=dict(color="#FF4B4B")
            ),
            row=2,
            col=1,
        )
        fig_dec.add_trace(
            go.Scatter(
                x=res.seasonal.index,
                y=res.seasonal,
                name="Seasonal",
                line=dict(color="#00E676"),
            ),
            row=3,
            col=1,
        )
        fig_dec.add_trace(
            go.Scatter(
                x=res.resid.index,
                y=res.resid,
                name="Residual",
                mode="markers",
                marker=dict(color="#FAFAFA", size=4),
            ),
            row=4,
            col=1,
        )

        fig_dec.update_layout(
            height=800,
            showlegend=False,
            template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_dec, use_container_width=True)

        st.markdown("""
        💡 **Key Insights (Decomposition):**
        * **The Trend Component:** Isolates the long-term, multi-year trajectory of the data, showing whether the overall baseline is naturally rising or falling over time.
        * **The Seasonal Component:** Extracts the purely cyclical, repeating annual patterns driven by the Earth's orbit and local monsoons.
        * **The Residual Component:** Represents anomalies, shocks, and irregular weather events that the model cannot explain.
        """)

        st.markdown("---")

        # 2. ACF & PACF Plots
        st.markdown(f"#### Autocorrelation Analysis: {zone_choice}")
        st.markdown(
            "Analyze the correlation of the time series with its own past values to identify AR (Auto-Regressive) and MA (Moving Average) parameters."
        )

        with plt.style.context("dark_background"):
            fig_acf, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

            fig_acf.patch.set_facecolor("#0E1117")
            ax1.set_facecolor("#0E1117")
            ax2.set_facecolor("#0E1117")

            series_for_acf = monthly_data[zone_choice].dropna()

            plot_acf(
                series_for_acf,
                ax=ax1,
                lags=36,
                color="#00D4B2",
                vlines_kwargs={"colors": "#00D4B2"},
            )
            ax1.set_title(f"ACF: {zone_choice}")
            ax1.tick_params(colors="white")

            plot_pacf(
                series_for_acf,
                ax=ax2,
                lags=36,
                method="ywm",
                color="#FF4B4B",
                vlines_kwargs={"colors": "#FF4B4B"},
            )
            ax2.set_title(f"PACF: {zone_choice}")
            ax2.tick_params(colors="white")

            plt.tight_layout()
            st.pyplot(fig_acf)

        st.markdown("""
        💡 **Key Insights (Autocorrelation):**
        * **Seasonality Confirmation (ACF):** Scalloped shapes or significant spikes at regular intervals (e.g., lag 12, 24) in the ACF plot confirm strong annual seasonality.
        * **Determining AR terms (PACF):** A sharp cut-off in the Partial Autocorrelation plot after a certain number of lags suggests the appropriate 'p' order for an Auto-Regressive model.
        * **Determining MA terms (ACF):** A sharp cut-off in the Autocorrelation plot suggests the appropriate 'q' order for a Moving Average model.
        """)

# ---------------------------------------------------------
# Page 3: K-Means Clustering
# ---------------------------------------------------------
elif page == "K-Means Clustering":
    st.title("K-Means Weather Clustering")
    st.markdown(
        "Discover hidden climate profiles by automatically grouping regions based on their Maximum Temperature and Maximum Wind Speed."
    )

    if filtered_weather.empty:
        st.stop()

    scaler_path = "Models/kmeans_scaler.pkl"
    model_path = "Models/kmeans_default.pkl"

    if os.path.exists(scaler_path) and os.path.exists(model_path):
        try:
            scaler = joblib.load(scaler_path)
            kmeans_model = joblib.load(model_path)

            cluster_features = ["temperature_2m_max", "windspeed_10m_max"]
            valid_data = filtered_weather.dropna(subset=cluster_features).copy()

            if not valid_data.empty:
                X_scaled = scaler.transform(valid_data[cluster_features])
                valid_data["Cluster"] = kmeans_model.predict(X_scaled)

                # Dynamic Naming
                cluster_summary = (
                    valid_data.groupby("Cluster")[cluster_features].mean().reset_index()
                )

                def generate_profile_name(temp, wind, cluster_id):
                    t_label = (
                        "Hot" if temp >= 32 else ("Warm" if temp >= 28 else "Mild")
                    )
                    w_label = (
                        "Windy" if wind >= 20 else ("Breezy" if wind >= 12 else "Calm")
                    )
                    return f"Profile {cluster_id}: {t_label}-{w_label}"

                cluster_summary["Cluster_Label"] = cluster_summary.apply(
                    lambda row: generate_profile_name(
                        row["temperature_2m_max"],
                        row["windspeed_10m_max"],
                        int(row["Cluster"]),
                    ),
                    axis=1,
                )

                name_map = dict(
                    zip(cluster_summary["Cluster"], cluster_summary["Cluster_Label"])
                )
                valid_data["Cluster_Label"] = valid_data["Cluster"].map(name_map)

                # Prediction Widget
                st.markdown("---")
                st.markdown("### 🔮 Predict Your Own Climate Profile")
                st.markdown(
                    "Enter hypothetical weather conditions to see which profile the machine learning model assigns them to."
                )

                with st.form("prediction_form"):
                    c1, c2, c3 = st.columns([2, 2, 1])
                    with c1:
                        user_temp = st.number_input(
                            "Max Temperature (°C)",
                            min_value=10.0,
                            max_value=50.0,
                            value=30.0,
                            step=0.1,
                        )
                    with c2:
                        user_wind = st.number_input(
                            "Max Wind Speed (km/h)",
                            min_value=0.0,
                            max_value=150.0,
                            value=25.0,
                            step=0.1,
                        )
                    with c3:
                        st.markdown("<br>", unsafe_allow_html=True)
                        submit_button = st.form_submit_button("Predict Profile")

                user_cluster_label = None
                if submit_button:
                    user_df = pd.DataFrame(
                        {
                            "temperature_2m_max": [user_temp],
                            "windspeed_10m_max": [user_wind],
                        }
                    )
                    user_scaled = scaler.transform(user_df)
                    user_cluster_id = kmeans_model.predict(user_scaled)[0]
                    user_cluster_label = name_map[user_cluster_id]
                    st.success(
                        f"⚡ The Machine Learning Model assigns these conditions to **{user_cluster_label}**!"
                    )

                # Cluster Scatter Map
                st.markdown("### 🎯 Climate Profile Distribution")
                fig_cluster = px.scatter(
                    valid_data,
                    x="temperature_2m_max",
                    y="windspeed_10m_max",
                    color="Cluster_Label",
                    hover_data=["city", "zone"],
                    labels={
                        "temperature_2m_max": "Max Temperature (°C)",
                        "windspeed_10m_max": "Max Wind Speed (km/h)",
                    },
                    template="plotly_dark",
                    color_discrete_sequence=px.colors.qualitative.Pastel,
                )

                if user_cluster_label is not None:
                    fig_cluster.add_trace(
                        go.Scatter(
                            x=[user_temp],
                            y=[user_wind],
                            mode="markers+text",
                            text=["📍 Your Input"],
                            textposition="top center",
                            textfont=dict(
                                color="#FF4B4B", size=14, family="Arial Black"
                            ),
                            marker=dict(
                                color="#FF4B4B",
                                size=20,
                                symbol="star",
                                line=dict(color="white", width=2),
                            ),
                            name="Your Prediction",
                            hovertemplate="Temp: %{x}°C<br>Wind: %{y} km/h<extra></extra>",
                        )
                    )

                fig_cluster.update_layout(
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    legend_title="Generated Profiles",
                    height=550,
                )
                st.plotly_chart(fig_cluster, use_container_width=True)

                st.markdown("""
                💡 **Key Insights:**
                * **Profile Segregation:** Distinct groupings reveal the primary weather archetypes in the dataset, completely bypassing manual classification.
                * **Interactive Assessment:** The prediction tool allows stakeholders to immediately classify hypothetical or forecasted weather events into established historical profiles.
                * **Extreme Outliers:** Points that sit on the fringes, far away from the dense cluster centers, represent extreme or anomalous weather events.
                """)

                st.markdown("---")
                st.markdown("### 📊 Cluster Characteristics")

                cols = st.columns(len(cluster_summary))
                for i, row in cluster_summary.iterrows():
                    with cols[i]:
                        st.markdown(f"**{row['Cluster_Label']}**")
                        st.metric(
                            label="Average Max Temp",
                            value=f"{row['temperature_2m_max']:.1f} °C",
                        )
                        st.metric(
                            label="Average Max Wind",
                            value=f"{row['windspeed_10m_max']:.1f} km/h",
                        )

        except Exception as e:
            st.error(f"Error applying the clustering model: {e}")
    else:
        st.warning(
            "⚠️ Pre-trained K-Means model or scaler not found. Please ensure your clustering script has successfully generated `kmeans_scaler.pkl` and `kmeans_default.pkl` inside your `Models` folder."
        )

        img_path = "Models/kmeans_optimization_chart.png"
        if os.path.exists(img_path):
            st.markdown("### 📈 Optimization Results (Static)")
            st.image(
                img_path,
                caption="Silhouette Score Optimization across K clusters.",
                use_container_width=True,
            )

# ---------------------------------------------------------
# Page 4: SARIMA Forecasts
# ---------------------------------------------------------
elif page == "SARIMA Forecasts":
    st.title("SARIMA Forecast Modeling")

    if filtered_weather.empty:
        st.stop()

    col1, col2 = st.columns(2)
    with col1:
        model_type = st.selectbox(
            "Target Variable:", ["Evapotranspiration", "Wind Speed"]
        )

    target_col = (
        "et0_fao_evapotranspiration"
        if model_type == "Evapotranspiration"
        else "windspeed_10m_max"
    )
    zone_col = "zone" if model_type == "Evapotranspiration" else "wind_zone"
    zones = filtered_weather[zone_col].dropna().unique()

    with col2:
        selected_zone = st.selectbox("Forecast Region:", zones)

    safe_zone_name = selected_zone.lower().replace(" ", "_")
    prefix = "et0" if model_type == "Evapotranspiration" else "wind"
    model_path = f"Models/sarima_{prefix}_{safe_zone_name}.pkl"

    series = (
        filtered_weather[filtered_weather[zone_col] == selected_zone]
        .groupby("time")[target_col]
        .mean()
        .resample("ME")
        .mean()
        .dropna()
    )

    if len(series) >= 24:
        train = series.iloc[:-12]
        test = series.iloc[-12:]

        fig_pred = go.Figure()

        fig_pred.add_trace(
            go.Scatter(
                x=train.index,
                y=train,
                mode="lines",
                name="Historical (Train)",
                line=dict(color="#5c6370"),
            )
        )
        fig_pred.add_trace(
            go.Scatter(
                x=test.index,
                y=test,
                mode="lines",
                name="Actuals (Test)",
                line=dict(color="#00D4B2", width=2),
            )
        )

        if os.path.exists(model_path):
            try:
                results = joblib.load(model_path)
                forecast_obj = results.get_forecast(steps=12)
                pred_mean = forecast_obj.predicted_mean
                conf_int = forecast_obj.conf_int()

                # Metrics
                mae = mean_absolute_error(test, pred_mean)
                rmse = np.sqrt(mean_squared_error(test, pred_mean))

                st.markdown(f"### Model Evaluation: {selected_zone}")
                met1, met2, met3 = st.columns([1, 1, 2])
                met1.metric(label="Mean Absolute Error (MAE)", value=f"{mae:.4f}")
                met2.metric(label="Root Mean Squared Error (RMSE)", value=f"{rmse:.4f}")

                st.markdown("---")

                # Forecast plotting
                fig_pred.add_trace(
                    go.Scatter(
                        x=conf_int.index.tolist() + conf_int.index[::-1].tolist(),
                        y=conf_int.iloc[:, 1].tolist()
                        + conf_int.iloc[:, 0][::-1].tolist(),
                        fill="toself",
                        fillcolor="rgba(255, 75, 75, 0.2)",
                        line=dict(color="rgba(255,255,255,0)"),
                        hoverinfo="skip",
                        showlegend=False,
                    )
                )

                fig_pred.add_trace(
                    go.Scatter(
                        x=pred_mean.index,
                        y=pred_mean,
                        mode="lines",
                        name="SARIMA Forecast",
                        line=dict(color="#FF4B4B", dash="dash", width=2),
                    )
                )

            except Exception as e:
                st.error(f"Error loading model: {e}")
        else:
            st.warning(
                f"Pre-trained model '{model_path}' not found. Showing historical split only."
            )

        fig_pred.update_layout(
            title=f"SARIMA Model vs Actuals",
            template="plotly_dark",
            hovermode="x unified",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.1)"),
            yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.1)"),
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
        )
        st.plotly_chart(fig_pred, use_container_width=True)

        st.markdown("""
        💡 **Key Insights:**
        * **Predictive Accuracy:** The MAE and RMSE metrics above quantify exactly how far off the SARIMA predictions were from the real-world test data. Lower values indicate better forecasting precision.
        * **Pattern Recognition:** The dashed forecast line demonstrates the model's ability to learn and replicate the seasonal peaks and troughs from the historical training set.
        * **Uncertainty Bounds:** The shaded red area represents the confidence interval. A wider band indicates periods where the model is less certain about weather volatility.
        """)
    else:
        st.error("Insufficient data points.")
