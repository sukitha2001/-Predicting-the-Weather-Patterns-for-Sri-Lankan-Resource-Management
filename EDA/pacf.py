from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


WEATHER = pd.read_csv(
    "../Data/Cleaned_SriLanka_Weather_Dataset.csv",
    parse_dates=["time"],
)

# Monthly Aggregation
evap_monthly = (
    WEATHER.groupby([pd.Grouper(key="time", freq="ME"), "zone"])[
        "et0_fao_evapotranspiration"
    ]
    .mean()
    .unstack()
)

zones = evap_monthly.columns
fig, axes = plt.subplots(len(zones), 2, figsize=(15, 5 * len(zones)))

for i, zone in enumerate(zones):
    series = evap_monthly[zone].dropna()

    # ACF Plot
    plot_acf(series, ax=axes[i, 0], lags=36)
    axes[i, 0].set_title(f"ACF: {zone} Evapotranspiration")

    # PACF Plot
    plot_pacf(series, ax=axes[i, 1], lags=36, method="ywm")
    axes[i, 1].set_title(f"PACF: {zone} Evapotranspiration")

plt.tight_layout()
plt.show()

# Monthly Aggregation
evap_monthly = (
    WEATHER.groupby([pd.Grouper(key="time", freq="ME"), "wind_zone"])[
        "windspeed_10m_max"
    ]
    .mean()
    .unstack()
)


zones = evap_monthly.columns
fig, axes = plt.subplots(len(zones), 2, figsize=(15, 5 * len(zones)))

for i, zone in enumerate(zones):
    series = evap_monthly[zone].dropna()

    # ACF Plot
    plot_acf(series, ax=axes[i, 0], lags=36)
    axes[i, 0].set_title(f"ACF: {zone} Windspeed")

    # PACF Plot
    plot_pacf(series, ax=axes[i, 1], lags=36, method="ywm")
    axes[i, 1].set_title(f"PACF: {zone} Windspeed")

plt.tight_layout()
plt.show()
