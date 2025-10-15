import pandas as pd 
df = pd.read_csv('/Users/nouromran/Documents/upWork/Clustering Analysis/Data/date=30_datasetid=6251d61ff86c620aad6a04cd970817d7_assetid=3251994d12744a45db1c47c0df1082bf_revid=8ad4c0af6ee521ab3b361a76ae4e90bb.csv' , sep='|')



df = df[df["VINTAGE"].between(2010, 2024)]
df["AS AT DATE"] = pd.to_datetime(df["AS AT DATE"].astype(str), format="%Y%m%d")
df.columns = df.columns.str.upper().str.strip()

# --- Sort by FUND ID → VINTAGE YEAR → AS AT DATE ---
sort_cols = [c for c in ["FUND ID", "VINTAGE", "AS AT DATE"] if c in df.columns]
df = df.sort_values(by=sort_cols, ascending=[True, True, True])

# Reset index after sorting
df = df.reset_index(drop=True)

drop_Again=["SOURCE", "BENCHMARK NAME","BENCHMARK NAME","PE: BUYOUT FUND SIZE","BENCHMARK ID","GEOGRAPHIC EXPOSURE","COMMITMENT (CCY)","FUND SERIES NAME",
            "SINGLE DEAL FUND","DOMESTIC ACCOUNTING STANDARD","VALUATION PRACTICES","LINKED CONTINUATION FUND",
            "CONTINUATION FUND","CONTINUATION FUND","QUARTILE","PREQIN QUARTILE RANK","FUND NAME","VINTAGE"]
df.drop(columns=drop_Again, inplace=True, errors="ignore")
# Save or return
df.to_csv("sorted_funds.csv", index=False)
print(f"✅ Sorted dataset saved → sorted_funds.csv ({df.shape[0]} rows, {df.shape[1]} cols)")