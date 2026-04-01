from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

WEATHER = pd.read_csv("../Data/Cleaned_SriLanka_Weather_Dataset.csv")

WEATHER["time"] = pd.to_datetime(WEATHER["time"])

# Monthly evaporation
evap_monthly = (
    WEATHER.groupby([pd.Grouper(key="time", freq="M"), "zone"])[
        "et0_fao_evapotranspiration"
    ]
    .mean()
    .unstack()
)

# Monthly wind
wind_monthly = (
    WEATHER.groupby([pd.Grouper(key="time", freq="M"), "wind_zone"])[
        "windspeed_10m_max"
    ]
    .mean()
    .unstack()
)


def plot_decomposition(data, title):
    for zone in data.columns:
        print(f"Decomposing {title} for {zone}...")

        res = seasonal_decompose(data[zone].dropna(), model="additive", period=12)
        fig = res.plot()
        fig.set_size_inches(12, 8)
        plt.suptitle(f"{title} Decomposition: {zone}", fontsize=16)
        plt.show()


plot_decomposition(evap_monthly, "Evaporation")
plot_decomposition(wind_monthly, "Wind Speed")
