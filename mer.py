import pandas as pd

# ============================================================
# 🧱 FUND MERGE + SORT + FILTER PIPELINE
# ============================================================

# --- 1️⃣ Load both datasets ---
funds_perf = pd.read_csv("filtered_funds.csv")
funds_meta = pd.read_csv("sorted_funds.csv")

# --- 2️⃣ Merge on FUND ID ---
merged = pd.merge(funds_perf, funds_meta, on="FUND ID", how="inner")

# --- 3️⃣ Sort by FUND ID (or any other column you prefer) ---

merged.to_csv("merged_funds.csv", index=False)
print("✅ Saved: merged_funds.csv")

# --- 5️⃣ Quick summary ---
