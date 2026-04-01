import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
import warnings

warnings.filterwarnings("ignore")

# 1. Setup and Data Loading
print("Loading cleaned dataset...")
try:
    WEATHER = pd.read_csv("../Data/Cleaned_SriLanka_Weather_Dataset.csv")
except FileNotFoundError:
    print("Error: Cleaned_SriLanka_Weather_Dataset.csv not found in data/ folder.")
    exit()

# Features used for clustering
features = ["temperature_2m_max", "windspeed_10m_max"]
X_raw = WEATHER[features]

# 2. Preprocessing
print("Scaling features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

# 3. Optimization Loop
results = []
best_silhouette = -1
best_k = 2
best_model = None

k_range = range(2, 9)  # Testing K from 2 to 8
print(f"Optimizing K-Means (testing K={list(k_range)})...")

for k in k_range:
    # Initialize and fit
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    # Calculate Metrics
    sil = silhouette_score(X_scaled, labels)
    db_index = davies_bouldin_score(X_scaled, labels)
    inertia = kmeans.inertia_

    results.append(
        {
            "k": k,
            "silhouette": sil,
            "db_index": db_index,
            "inertia": inertia,
            "model": kmeans,
        }
    )

    print(f"  K={k} | Silhouette: {sil:.4f} | DB Index: {db_index:.4f}")

    # Logic to decide the "Best" model
    # Higher Silhouette score generally indicates better defined clusters
    if sil > best_silhouette:
        best_silhouette = sil
        best_k = k
        best_model = kmeans

# 4. Final Evaluation Report
print("\n" + "=" * 50)
print(f"🏆 OPTIMIZATION COMPLETE: BEST K = {best_k}")
print(f"Selected based on Highest Silhouette Score: {best_silhouette:.4f}")
print("=" * 50)

# 5. Exporting the Winning Model
os.makedirs("models", exist_ok=True)

# Save the scaler (ESSENTIAL for the frontend to work)
joblib.dump(scaler, "../Models/kmeans_scaler.pkl")

# Save the best model
joblib.dump(best_model, "../Models/kmeans_default.pkl")

print(f"\n✅ Scaler and Best Model (K={best_k}) exported to models/ folder.")

# 6. (Optional) Generate a summary plot for your report
plt.figure(figsize=(10, 5))
ks = [r["k"] for r in results]
sils = [r["silhouette"] for r in results]

plt.plot(ks, sils, "bo-", linewidth=2)
plt.axvline(x=best_k, color="r", linestyle="--", label=f"Best K={best_k}")
plt.title("Silhouette Score by Cluster Count")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("../Models/kmeans_optimization_chart.png")
print("📊 Optimization chart saved to ../Models/kmeans_optimization_chart.png")
