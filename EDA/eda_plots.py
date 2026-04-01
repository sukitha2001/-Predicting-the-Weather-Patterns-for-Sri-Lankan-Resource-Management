import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

WEATHER = pd.read_csv("../Data/Cleaned_SriLanka_Weather_Dataset.csv")

WEATHER_with_wind_dir = WEATHER.copy()

# Wind direction


def plot_wind_rose(data, title, color="skyblue"):

    num_bins = 16
    bins = np.linspace(0, 360, num_bins + 1)
    bin_centers = np.deg2rad(bins[:-1] + np.diff(bins) / 2)

    counts, _ = np.histogram(data["winddirection_10m_dominant"], bins=bins)
    frequency = counts / counts.sum() if counts.sum() > 0 else counts

    fig, ax = plt.subplots(figsize=(8, 6), subplot_kw=dict(polar=True))
    ax.bar(
        bin_centers,
        frequency,
        width=np.deg2rad(360 / num_bins),
        bottom=0.0,
        color=color,
        edgecolor="black",
        alpha=0.7,
    )

    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)  # Clockwise
    ax.set_xticks(np.deg2rad([0, 45, 90, 135, 180, 225, 270, 315]))
    ax.set_xticklabels(["N", "NE", "E", "SE", "S", "SW", "W", "NW"])

    max_freq = max(frequency) if len(frequency) > 0 else 0.1
    ticks = np.linspace(0, max_freq, 5)
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{y * 100:.1f}%" for y in ticks])

    plt.title(title, pad=20, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()


plot_wind_rose(
    WEATHER_with_wind_dir, "Overall Dominant Wind Direction Frequency (Sri Lanka)"
)

colors = {
    "Zone 1": "#3498db",
    "Zone 2": "#e67e22",
    "Zone 3": "#2ecc71",
    "Unknown": "#95a5a6",
}

for w_zone in WEATHER_with_wind_dir["wind_zone"].unique():
    subset = WEATHER_with_wind_dir[WEATHER_with_wind_dir["wind_zone"] == w_zone]
    if not subset.empty:
        plot_wind_rose(
            subset,
            f"Wind Direction Frequency: {w_zone}",
            color=colors.get(w_zone, "skyblue"),
        )


# ANALYZING MONTHLY WIND SPEED TRENDS BY ZONE


def plot_monthly_wind_speed(df):

    df["time"] = pd.to_datetime(df["time"])
    df["month"] = df["time"].dt.month

    monthly_stats = (
        df.groupby(["wind_zone", "month"])["windspeed_10m_max"].mean().reset_index()
    )

    plt.figure(figsize=(12, 6))

    sns.lineplot(
        data=monthly_stats,
        x="month",
        y="windspeed_10m_max",
        hue="wind_zone",
        marker="o",
        linewidth=2.5,
    )

    plt.title("Average Monthly Wind Speed by Wind Zone", fontsize=15, fontweight="bold")
    plt.xlabel("Month", fontsize=12)
    plt.ylabel("Average Wind Speed (m/s)", fontsize=12)
    plt.xticks(
        range(1, 13),
        [
            "Jan",
            "Feb",
            "Mar",
            "Apr",
            "May",
            "Jun",
            "Jul",
            "Aug",
            "Sep",
            "Oct",
            "Nov",
            "Dec",
        ],
    )
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(title="Wind Zone", bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()
    plt.show()


plot_monthly_wind_speed(WEATHER_with_wind_dir)


# Evaporation by City


def map_evaporation_by_city(df):

    city_stats = (
        df.groupby(["zone", "city"])["et0_fao_evapotranspiration"].mean().reset_index()
    )

    city_stats = city_stats.sort_values(
        by=["zone", "et0_fao_evapotranspiration"], ascending=[True, False]
    )

    plt.figure(figsize=(12, 10))

    sns.barplot(
        data=city_stats,
        x="et0_fao_evapotranspiration",
        y="city",
        hue="zone",
        dodge=False,
        palette={
            "Wet Zone": "#2ecc71",
            "Intermediate Zone": "#f1c40f",
            "Dry Zone": "#e74c3c",
        },
    )

    plt.title(
        "Average Daily Evapotranspiration (ET0) by City & Zone",
        fontsize=16,
        fontweight="bold",
    )
    plt.xlabel("Average ET0 (mm/day)", fontsize=12)
    plt.ylabel("City", fontsize=12)
    plt.grid(axis="x", linestyle="--", alpha=0.6)
    plt.legend(title="Climatic Zone", loc="lower right")

    for index, value in enumerate(city_stats["et0_fao_evapotranspiration"]):
        plt.text(value + 0.05, index, f"{value:.2f}", va="center", fontsize=9)

    plt.tight_layout()
    plt.show()

    return city_stats


avg_evap_stats = map_evaporation_by_city(WEATHER)


weather_cols = ["et0_fao_evapotranspiration", "precipitation_sum"]

WEATHER["time"] = pd.to_datetime(WEATHER["time"])

zonal_daily = (
    WEATHER.groupby(["time", "zone"])[weather_cols].mean(numeric_only=True).unstack()
)

zonal_daily.columns = [f"{col[0]}_{col[1]}" for col in zonal_daily.columns]


annual_rain = zonal_daily.resample("YE").sum()


baseline_rain = annual_rain["precipitation_sum_Wet Zone"].mean()


plt.figure(figsize=(14, 7))


target_years = [2016, 2017, 2020, 2023]
colors = [
    "#e74c3c" if year in target_years else "#3498db" for year in annual_rain.index.year
]

bars = plt.bar(
    annual_rain.index.year,
    annual_rain["precipitation_sum_Wet Zone"],
    color=colors,
    alpha=0.8,
    edgecolor="black",
    linewidth=0.5,
)

plt.axhline(
    baseline_rain,
    color="black",
    linestyle="--",
    label=f"Baseline Avg: {baseline_rain:.0f}mm",
)


plt.title("Wet Zone Annual Rainfall: Deficit Analysis (2010-2025)", fontsize=15)
plt.ylabel("Total Annual Rainfall (mm)", fontsize=12)
plt.xlabel("Year", fontsize=12)
plt.xticks(annual_rain.index.year, rotation=45)
plt.legend(loc="upper left")
plt.grid(axis="y", alpha=0.2)


for bar, year in zip(bars, annual_rain.index.year):
    if year in target_years:
        val = bar.get_height()
        perc = (val / baseline_rain - 1) * 100
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            val + 20,
            f"{perc:+.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            color="#c0392b",
        )

plt.tight_layout()
plt.show()
