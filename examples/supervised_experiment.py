import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from mic_dp.core import (
    noise_scaling_MIC, noise_scaling_pearson, noise_scaling_mahalanobis_distances,
    feature_selection, calculate_sensitivity, correlated_dp_gaussian,
    mean_absolute_error, cluster_and_evaluate, calculate_ari, calculate_v_measure
)

# Set the correct path to the repository root
ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

# Set IEEE-style plotting
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 8,
    'axes.labelsize': 8,
    'axes.titlesize': 8,
    'legend.fontsize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'figure.dpi': 300
})

# Global parameters
PERCENTAGE = 1
DELTA = 1e-5
AMPLIFICATION_FACTOR = 5
EPSILONS = np.arange(0.1, 1, 0.1)

# =====================================
# Load Adult Dataset (Supervised)
# =====================================
print("Loading dataset...")
dataset_path = os.path.join(ROOT, 'data', 'sample_data.csv')
print(f"Looking for dataset at: {dataset_path}")
df = pd.read_csv(dataset_path)
df.dropna(inplace=True)
X = df.select_dtypes(include=['number'])
X_norm = pd.DataFrame(MinMaxScaler().fit_transform(X), columns=X.columns)
y = df['Daily_Revenue'].astype('category').cat.codes

# ==============================
# Feature Selection & Noise Scales
# ==============================
features = X_norm.copy()
target = y
noise_MIC = noise_scaling_MIC(target, features, AMPLIFICATION_FACTOR)
noise_pearson = noise_scaling_pearson(target, features, AMPLIFICATION_FACTOR)
noise_mahal = noise_scaling_mahalanobis_distances(target, features, AMPLIFICATION_FACTOR)
noise_base = {f: AMPLIFICATION_FACTOR for f in features}

selected = feature_selection(PERCENTAGE, features, noise_MIC)
features = features[selected]
noise_MIC = {k: noise_MIC[k] for k in selected}
noise_pearson = {k: noise_pearson[k] for k in selected}
noise_mahal = {k: noise_mahal[k] for k in selected}
noise_base = {k: noise_base[k] for k in selected}
sensitivity = calculate_sensitivity(features)

# =====================================
# Evaluate MAE vs Epsilon
# =====================================
mic_mae, pearson_mae, mahal_mae, base_mae = [], [], [], []
mic_pred_mae, pearson_pred_mae, mahal_pred_mae, base_pred_mae = [], [], [], []
model = RandomForestRegressor()

for eps in EPSILONS:
    for name, noise in zip(
        ['MIC', 'Pearson', 'Mahal', 'Base'],
        [noise_MIC, noise_pearson, noise_mahal, noise_base]
    ):
        noisy = correlated_dp_gaussian(features.copy(), noise, sensitivity, eps, DELTA)
        model.fit(noisy, target)
        pred = model.predict(noisy)
        feature_mae = np.mean(np.abs(noisy - features))
        pred_mae = mean_absolute_error(target, pred)
        print(f"Eps={eps:.2f}, {name}: Feature MAE={feature_mae:.4f}, Prediction MAE={pred_mae:.4f}")
        locals()[f"{name.lower()}_mae"].append(feature_mae)
        locals()[f"{name.lower()}_pred_mae"].append(pred_mae)

mic_mae = list(gaussian_filter1d(mic_mae, sigma=2))
pearson_mae = list(gaussian_filter1d(pearson_mae,2))
mahal_mae = list(gaussian_filter1d(mahal_mae, sigma=2))
base_mae = list(gaussian_filter1d(base_mae, sigma=2))


# Plot feature MAE
plt.figure(figsize=(3.5, 3.5))
plt.plot(EPSILONS, mic_mae, label="MIC", linestyle='-')
plt.plot(EPSILONS, pearson_mae, label="Pearson", linestyle='--')
plt.plot(EPSILONS, mahal_mae, label="Mahalanobis", linestyle=':')
plt.plot(EPSILONS, base_mae, label="Baseline", linestyle='-.')
plt.xlabel("Privacy budget ε")
plt.ylabel("MAE (Features)")
plt.grid(True, linestyle=":")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(ROOT, 'MAE.png'))
print(f"Feature MAE plot saved to {os.path.join(ROOT, 'MAE.png')}")

mic_pred_mae = list(gaussian_filter1d(mic_pred_mae, sigma=2))
pearson_pred_mae = list(gaussian_filter1d(pearson_pred_mae,2))
mahal_pred_mae = list(gaussian_filter1d(mahal_pred_mae, sigma=2))
base_pred_mae = list(gaussian_filter1d(base_pred_mae, sigma=2))

# Plot prediction MAE
plt.figure(figsize=(3.5, 3.5))
plt.plot(EPSILONS, mic_pred_mae, label="MIC", linestyle='-')
plt.plot(EPSILONS, pearson_pred_mae, label="Pearson", linestyle='--')
plt.plot(EPSILONS, mahal_pred_mae, label="Mahalanobis", linestyle=':')
plt.plot(EPSILONS, base_pred_mae, label="Baseline", linestyle='-.')
plt.xlabel("Privacy budget ε")
plt.ylabel("MAE (Prediction)")
plt.grid(True, linestyle=":")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(ROOT, 'MAE_pred.png'))
print(f"Prediction MAE plot saved to {os.path.join(ROOT, 'MAE_pred.png')}")

print("Experiment completed successfully!")
