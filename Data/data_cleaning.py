import numpy as np
import pandas as pd

# used for piping
from dfply import *


import warnings

warnings.filterwarnings("ignore")


# Importing Data
data_path = "../Data/SriLanka_Weather_Dataset.csv"
df = pd.read_csv(data_path)
df["time"] = pd.to_datetime(df["time"])
print(f"Shape: {df.shape}, Cities: {df['city'].nunique()}")
print(f"Date Range: {df['time'].min()} to {df['time'].max()}")

df = df >> drop(
    "weathercode", "sunrise", "sunset", "country", "rain_sum", "snowfall_sum"
)

WEATHER = df.copy()

WEATHER["time"] = pd.to_datetime(WEATHER["time"])

WEATHER = WEATHER.sort_values(by=["city", "time"]).reset_index(drop=True)

cleaned_WEATHER = []

for city_name, group in WEATHER.groupby("city"):
    group = group.copy()

    numeric_cols = group.select_dtypes(include=[np.number]).columns

    group[numeric_cols] = group[numeric_cols].interpolate(method="linear")
    group[numeric_cols] = group[numeric_cols].ffill().bfill()

    cleaned_WEATHER.append(group)

WEATHER = pd.concat(cleaned_WEATHER).reset_index(drop=True)

zone_map = {
    # --- WET ZONE ---
    "Colombo": "Wet Zone",
    "Mount Lavinia": "Wet Zone",
    "Kesbewa": "Wet Zone",
    "Moratuwa": "Wet Zone",
    "Maharagama": "Wet Zone",
    "Ratnapura": "Wet Zone",
    "Galle": "Wet Zone",
    "Athurugiriya": "Wet Zone",
    "Weligama": "Wet Zone",
    "Matara": "Wet Zone",
    "Kolonnawa": "Wet Zone",
    "Gampaha": "Wet Zone",
    "Kalutara": "Wet Zone",
    "Bentota": "Wet Zone",
    "Mabole": "Wet Zone",
    "Hatton": "Wet Zone",
    "Oruwala": "Wet Zone",
    "Negombo": "Wet Zone",
    "Sri Jayewardenepura Kotte": "Wet Zone",
    "Kandy": "Wet Zone",
    # --- DRY ZONE ---
    "Jaffna": "Dry Zone",
    "Mannar": "Dry Zone",
    "Puttalam": "Dry Zone",
    "Trincomalee": "Dry Zone",
    "Kalmunai": "Dry Zone",
    "Hambantota": "Dry Zone",
    # --- INTERMEDIATE ZONE ---
    "Kurunegala": "Intermediate Zone",
    "Pothuhera": "Intermediate Zone",
    "Matale": "Intermediate Zone",
    "Badulla": "Intermediate Zone",
}

WEATHER["zone"] = WEATHER["city"].map(zone_map)

# Define the mapping of cities to their respective Wind Zones
wind_zone_map = {
    # Zone I: Northern & Eastern Coasts
    "Jaffna": "Zone I",
    "Trincomalee": "Zone I",
    "Kalmunai": "Zone I",  # Eastern Coast
    # Zone II: Intermediate areas
    "Puttalam": "Zone II",
    "Mannar": "Zone II",
    "Kurunegala": "Zone II",  # Intermediate climatic zone
    "Pothuhera": "Zone II",  # Intermediate climatic zone
    "Matale": "Zone II",  # Intermediate climatic zone
    # Zone III: Southern & Western areas
    "Colombo": "Zone III",
    "Kandy": "Zone III",
    "Galle": "Zone III",
    "Mount Lavinia": "Zone III",
    "Kesbewa": "Zone III",
    "Moratuwa": "Zone III",
    "Maharagama": "Zone III",
    "Ratnapura": "Zone III",
    "Athurugiriya": "Zone III",
    "Weligama": "Zone III",
    "Matara": "Zone III",
    "Kolonnawa": "Zone III",
    "Gampaha": "Zone III",
    "Kalutara": "Zone III",
    "Bentota": "Zone III",
    "Mabole": "Zone III",
    "Hatton": "Zone III",
    "Oruwala": "Zone III",
    "Negombo": "Zone III",
    "Sri Jayewardenepura Kotte": "Zone III",
    "Hambantota": "Zone III",  # Southern Coast
    "Badulla": "Zone III",  # Hilly/Southern Central, aligns with Zone III
}


WEATHER["wind_zone"] = WEATHER["city"].map(wind_zone_map)

WEATHER["wind_zone"] = WEATHER["wind_zone"].fillna("Unknown")

# Save the cleaned, sorted, and mapped data for the backend to use
WEATHER.to_csv("../Data/Cleaned_SriLanka_Weather_Dataset.csv", index=False)
