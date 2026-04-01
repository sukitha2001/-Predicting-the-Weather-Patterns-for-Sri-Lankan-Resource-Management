from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pmdarima as pm
import warnings
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore")

WEATHER = pd.read_csv("../Data/Cleaned_SriLanka_Weather_Dataset.csv")

WEATHER["time"] = pd.to_datetime(WEATHER["time"])


def analyze_weather_variable(df, zone_name, target_col, zone_col):

    print(f"\n" + "=" * 60)
    print(f"ANALYZING: {zone_name} | VARIABLE: {target_col}")
    print("=" * 60)

    # Filter and Resample
    series = (
        df[df[zone_col] == zone_name]
        .groupby("time")[target_col]
        .mean()
        .resample("ME")
        .mean()
        .dropna()
    )

    if len(series) < 24:
        print(f"Skipping {zone_name}: Not enough data points for seasonal SARIMA.")
        return None

    # Train-Test Split (Last 12 months for testing)
    train = series.iloc[:-12]
    test = series.iloc[-12:]

    # Automated Parameter Selection
    print(f"Optimizing SARIMA for {target_col}...")
    stepwise_model = pm.auto_arima(
        train,
        seasonal=True,
        m=12,
        d=1,
        D=1,
        trace=False,
        error_action="ignore",
        suppress_warnings=True,
        stepwise=True,
    )

    # Fit Final Model
    model = SARIMAX(
        train,
        order=stepwise_model.order,
        seasonal_order=stepwise_model.seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    results = model.fit(disp=False)

    # Forecast & Evaluation
    forecast_obj = results.get_forecast(steps=12)
    pred_mean = forecast_obj.predicted_mean
    conf_int = forecast_obj.conf_int()

    mae = mean_absolute_error(test, pred_mean)
    rmse = np.sqrt(mean_squared_error(test, pred_mean))

    print(f"Optimal Model: {stepwise_model.order}x{stepwise_model.seasonal_order}")
    print(f"MAE: {mae:.4f} | RMSE: {rmse:.4f}")

    # Visualization
    plt.figure(figsize=(12, 5))
    plt.plot(train.index, train, label="Historical (Train)", color="#34495e", alpha=0.5)
    plt.plot(test.index, test, label="Actual (Test)", color="#27ae60", linewidth=2)
    plt.plot(
        pred_mean.index,
        pred_mean,
        label="SARIMA Forecast",
        color="#e67e22",
        linestyle="--",
    )
    plt.fill_between(
        conf_int.index,
        conf_int.iloc[:, 0],
        conf_int.iloc[:, 1],
        color="#e67e22",
        alpha=0.1,
    )

    plt.title(f"{target_col} Forecast: {zone_name}")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    return results


# ---------------------------------------------------------
# 1. Analyze and Export Evapotranspiration Models
# ---------------------------------------------------------
evap_target = "et0_fao_evapotranspiration"
for zone in WEATHER["zone"].unique():
    if pd.notna(zone):
        # Capture the model returned by your function
        model_result = analyze_weather_variable(WEATHER, zone, evap_target, "zone")

        if model_result is not None:
            # Format the filename (e.g., 'Dry Zone' becomes 'dry_zone')
            safe_zone_name = zone.lower().replace(" ", "_")
            filename = f"../Models/sarima_et0_{safe_zone_name}.pkl"

            # Export the model
            joblib.dump(model_result, filename)
            print(f"✅ Successfully exported: {filename}")

# ---------------------------------------------------------
# 2. Analyze and Export Wind Speed Models
# ---------------------------------------------------------
wind_target = "windspeed_10m_max"
for zone in WEATHER["wind_zone"].unique():
    if pd.notna(zone):
        # Capture the model returned by your function
        model_result = analyze_weather_variable(WEATHER, zone, wind_target, "wind_zone")

        if model_result is not None:
            # Format the filename
            safe_zone_name = zone.lower().replace(" ", "_")
            filename = f"../Models/sarima_wind_{safe_zone_name}.pkl"

            # Export the model
            joblib.dump(model_result, filename)
            print(f"✅ Successfully exported: {filename}")
