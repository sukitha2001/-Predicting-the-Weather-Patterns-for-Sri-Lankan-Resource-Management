# Create a virtual environment
python -m venv .venv

# Activate the virtual environment
# For macOS/Linux:
source .venv/bin/activate
# For Windows:
.venv\Scripts\activate

# To Run
pip install -r requirements.txt
streamlit run app.py


## 📂 Directory Structure
To ensure the dashboard loads all models, assets, and data correctly, maintain the following hierarchy:

```text
SriLanka-Weather-Forecasting/
├── .streamlit/
│   └── config.toml          # Forced dark theme & branding colors
├── Assets/
│   └── logo.png             # Project minimalist logo
├── Data/
│   └── Cleaned_SriLanka_Weather_Dataset.csv
├── Models/
│   ├── kmeans_scaler.pkl    # Exported Scaler for ML mapping
│   ├── kmeans_default.pkl   # Winning K-Means clustering model
│   ├── sarima_et0_... .pkl  # Trained SARIMA models (Evapotranspiration)
│   └── sarima_wind_... .pkl # Trained SARIMA models (Wind)
├── app.py                   # Main Streamlit Dashboard Script
└── requirements.txt         # Project dependencies

