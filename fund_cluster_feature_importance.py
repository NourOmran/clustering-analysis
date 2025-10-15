import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif

# ============================================================
# 1️⃣ Load and Prepare Data
# ============================================================
df = pd.read_csv("merged_funds.csv")

cols = [
    "FUND ID", "VINTAGE YEAR", "STRATEGY", "CORE INDUSTRIES",
    "NET MULTIPLE (X)", "NET IRR (%)", "DPI (%)", "RVPI (%)",
    "CALLED (%)", "CALLED", "DISTRIBUTED", "REMAINING"
]
df = df[cols].copy()

numeric_cols = [
    "NET MULTIPLE (X)", "NET IRR (%)", "DPI (%)", "RVPI (%)",
    "CALLED (%)", "CALLED", "DISTRIBUTED", "REMAINING"
]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=["NET MULTIPLE (X)", "NET IRR (%)", "DPI (%)", "RVPI (%)"], how="all")

df["FUND_KEY"] = (
    df["FUND ID"].astype(str) + "_" +
    df["VINTAGE YEAR"].astype(str) + "_" +
    df["STRATEGY"].astype(str)
)

agg_df = df.groupby(["FUND_KEY", "CORE INDUSTRIES"])[numeric_cols].mean().reset_index()

# One-Hot encode CORE INDUSTRIES
encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
industry_encoded = encoder.fit_transform(agg_df[["CORE INDUSTRIES"]])
industry_cols = [f"IND_{c.split('_')[-1]}" for c in encoder.get_feature_names_out()]
industry_df = pd.DataFrame(industry_encoded, columns=industry_cols, index=agg_df.index)

features_df = pd.concat([agg_df.drop(columns=["CORE INDUSTRIES"]), industry_df], axis=1)
feature_cols = numeric_cols + industry_cols

# ============================================================
# 2️⃣ Scale Features
# ============================================================
X = features_df[feature_cols].fillna(0)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ============================================================
# 3️⃣ Final Clustering with k = 14
# ============================================================
best_k = 14
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
features_df["CLUSTER"] = kmeans.fit_predict(X_scaled)

sil = silhouette_score(X_scaled, features_df["CLUSTER"])
db = davies_bouldin_score(X_scaled, features_df["CLUSTER"])
print(f"✅ Final clustering with k={best_k}: Silhouette={sil:.3f} | Davies–Bouldin={db:.3f}")

# ============================================================
# 4️⃣ PCA Visualization
# ============================================================
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(20,20))
plt.scatter(X_pca[:,0], X_pca[:,1], c=features_df["CLUSTER"], cmap='tab20', s=40, alpha=0.8)
plt.title(f"PCA Projection of Fund Clusters (k={best_k})")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar(label="Cluster ID")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()

# ============================================================
# 5️⃣ ANOVA F-Statistics (Feature Discrimination Power)
# ============================================================
F_values, p_values = f_classif(X_scaled, features_df["CLUSTER"])
anova_results = pd.DataFrame({
    "Feature": feature_cols,
    "F_Score": F_values,
    "p_Value": p_values
}).sort_values("F_Score", ascending=False)

print("\n🔍 Top 10 Features by ANOVA F-Score:")
print(anova_results.head(10))

# ============================================================
# 6️⃣ RandomForest Feature Importances
# ============================================================
rf = RandomForestClassifier(
    n_estimators=500, random_state=42, class_weight="balanced", n_jobs=-1
)
rf.fit(X_scaled, features_df["CLUSTER"])
rf_importance = pd.DataFrame({
    "Feature": feature_cols,
    "RF_Importance": rf.feature_importances_
}).sort_values("RF_Importance", ascending=False)

print("\n🌲 Top 10 Features by RandomForest Importance:")
print(rf_importance.head(10))

# ============================================================
# 7️⃣ Merge & Export Feature Importance Summary
# ============================================================
importance_summary = pd.merge(
    anova_results, rf_importance, on="Feature", how="outer"
).fillna(0)

importance_summary["Combined_Rank"] = (
    importance_summary["F_Score"].rank(ascending=False) +
    importance_summary["RF_Importance"].rank(ascending=False)
)

importance_summary = importance_summary.sort_values("Combined_Rank")
importance_summary.to_csv("feature_importance_summary.csv", index=False)
print("\n📂 Exported feature_importance_summary.csv")

# ============================================================
# 8️⃣ Optional: Visualize Top Feature Importances
# ============================================================
top_features = importance_summary.head(10)
plt.figure(figsize=(8,5))
plt.barh(top_features["Feature"], top_features["RF_Importance"], color="teal", alpha=0.7)
plt.gca().invert_yaxis()
plt.title("Top 10 Features Driving Clusters (RandomForest Importance)")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.show()
