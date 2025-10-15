import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

# ============================================================
# 1️⃣ LOAD DATA
# ============================================================
df = pd.read_csv("merged_funds.csv")

# Select relevant columns
cols = [
    "FUND ID", "VINTAGE YEAR", "STRATEGY", "CORE INDUSTRIES",
    "NET MULTIPLE (X)", "NET IRR (%)", "DPI (%)", "RVPI (%)",
    "CALLED (%)", "CALLED", "DISTRIBUTED", "REMAINING"
]
df = df[cols].copy()

# ============================================================
# 2️⃣ CLEAN NUMERIC COLUMNS
# ============================================================
numeric_cols = [
    "NET MULTIPLE (X)", "NET IRR (%)", "DPI (%)", "RVPI (%)",
    "CALLED (%)", "CALLED", "DISTRIBUTED", "REMAINING"
]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=["NET MULTIPLE (X)", "NET IRR (%)", "DPI (%)", "RVPI (%)"], how="all")

# Create unique key for each fund instance
df["FUND_KEY"] = (
    df["FUND ID"].astype(str) + "_" +
    df["VINTAGE YEAR"].astype(str) + "_" +
    df["STRATEGY"].astype(str)
)

# ============================================================
# 3️⃣ AGGREGATE PER FUND INSTANCE
# ============================================================
agg_df = df.groupby(["FUND_KEY", "CORE INDUSTRIES"])[numeric_cols].mean().reset_index()

# ============================================================
# 4️⃣ ENCODE CORE INDUSTRIES
# ============================================================
encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
industry_encoded = encoder.fit_transform(agg_df[["CORE INDUSTRIES"]])
industry_cols = [f"IND_{c.split('_')[-1]}" for c in encoder.get_feature_names_out()]
industry_df = pd.DataFrame(industry_encoded, columns=industry_cols, index=agg_df.index)

# Merge encoded industries
features_df = pd.concat([agg_df.drop(columns=["CORE INDUSTRIES"]), industry_df], axis=1)

# ============================================================
# 5️⃣ SCALE FEATURES
# ============================================================
feature_cols = numeric_cols + industry_cols
X = features_df[feature_cols].fillna(0)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled
# ============================================================
# 6️⃣ TEST MULTIPLE K VALUES (ELBOW + SILHOUETTE)
# ============================================================
inertia = []
silhouette_scores = []
K_range = range(2,50)

print("🔎 Testing different k values...")
for k in K_range:
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(X_scaled)
    inertia.append(model.inertia_)
    sil = silhouette_score(X_scaled, labels)
    silhouette_scores.append(sil)
    print(f"k={k:<2} | Inertia={model.inertia_:.2f} | Silhouette={sil:.3f}")

# Plot curves
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
ax[0].plot(K_range, inertia, 'o-', color='royalblue')
ax[0].set_title("Elbow Method (Within-Cluster SSE)")
ax[0].set_xlabel("Number of Clusters (k)")
ax[0].set_ylabel("Inertia (SSE)")
ax[0].grid(True, linestyle="--", alpha=0.5)

ax[1].plot(K_range, silhouette_scores, 'o-', color='darkorange')
ax[1].set_title("Silhouette Score by k")
ax[1].set_xlabel("Number of Clusters (k)")
ax[1].set_ylabel("Silhouette Score")
ax[1].grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.show()

# Determine best k by highest silhouette score
best_k = K_range[int(np.argmax(silhouette_scores))]
print(f"\n🏆 Best k (by Silhouette) = {best_k} with score {max(silhouette_scores):.3f}")

# ============================================================
# 7️⃣ FINAL CLUSTERING WITH BEST k
# ============================================================
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
features_df["CLUSTER"] = kmeans.fit_predict(X_scaled)

# Evaluate cluster compactness and separation
sil = silhouette_score(X_scaled, features_df["CLUSTER"])
db = davies_bouldin_score(X_scaled, features_df["CLUSTER"])
print(f"\n✅ Final clustering done — Silhouette: {sil:.3f} | Davies–Bouldin: {db:.3f}")

# ============================================================
# 8️⃣ SAVE OUTPUTS
# ============================================================
features_df.to_csv("clustered_funds_with_industries.csv", index=False)
print("📂 Exported clustered_funds_with_industries.csv")
print(f"✅ Final dataset saved → clustered_funds_with_industries.csv ({features_df.shape[0]} rows, {features_df.shape[1]} cols)")