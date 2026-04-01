# Group by time and zone, then calculate the mean et0_fao_evapotranspiration
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

WEATHER = pd.read_csv(
    "/Users/sukitharathnayake/CodeRepo/ST3011_Individual/Data/Cleaned_SriLanka_Weather_Dataset.csv"
)
eva_by_zone = (
    WEATHER.groupby(["time", "zone"])["et0_fao_evapotranspiration"].mean().reset_index()
)
# Evaporation by zone over time

# Apply a rolling mean for smoothing within each zone
eva_by_zone_smoothed = eva_by_zone.groupby("zone")[
    "et0_fao_evapotranspiration"
].transform(lambda x: x.rolling(window=30, min_periods=1).mean())
eva_by_zone["et0_fao_evapotranspiration_smoothed"] = eva_by_zone_smoothed

plt.figure(figsize=(16, 8))
sns.lineplot(
    data=eva_by_zone, x="time", y="et0_fao_evapotranspiration_smoothed", hue="zone"
)
plt.title(
    "Smoothed Time Series of Mean et0_fao_evapotranspiration by Climatic Zone (30-day rolling mean)"
)
plt.xlabel("Date")
plt.ylabel("Smoothed Mean et0_fao_evapotranspiration")
plt.grid(True)
plt.legend(title="Zone")
plt.tight_layout()
plt.show()

# Wind speed by zone over time

# Group by time and wind_zone, then calculate the mean windspeed_10m_max
wind_speed_by_zone_agg = (
    WEATHER.groupby(["time", "wind_zone"])["windspeed_10m_max"].mean().reset_index()
)

# Apply a rolling mean for smoothing within each wind zone

wind_speed_by_zone_agg["windspeed_10m_max_smoothed"] = wind_speed_by_zone_agg.groupby(
    "wind_zone"
)["windspeed_10m_max"].transform(lambda x: x.rolling(window=30, min_periods=1).mean())

plt.figure(figsize=(16, 8))
sns.lineplot(
    data=wind_speed_by_zone_agg,
    x="time",
    y="windspeed_10m_max_smoothed",
    hue="wind_zone",
)
plt.title(
    "Smoothed Time Series of Mean Maximum Wind Speed by Wind Zone (30-day rolling mean)"
)
plt.xlabel("Date")
plt.ylabel("Smoothed Mean Maximum Wind Speed (m/s)")
plt.grid(True)
plt.legend(title="Wind Zone")
plt.tight_layout()
plt.show()
