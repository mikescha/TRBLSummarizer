import pandas as pd
import numpy as np

# ---- CONFIG: update these paths as needed ----
WEATHER_PATH = "weather_history_for_temperature_study.csv"
ATTEMPTS_PATH = "TRBL Analysis tracking for temperature study.csv"
OUTPUT_PATH = "hatch_temp_summary.csv"

# ---- LOAD DATA ----
weather = pd.read_csv(WEATHER_PATH)
attempts = pd.read_csv(ATTEMPTS_PATH)

# Parse dates
weather["date"] = pd.to_datetime(weather["date"], errors="coerce")

# Treat nestling_onset as hatching date
# (change this column name if you later use a different hatch-date field)
attempts["nestling_onset"] = pd.to_datetime(
    attempts["nestling_onset"], errors="coerce"
)

rows = []

for _, r in attempts.iterrows():
    hatch = r["nestling_onset"]
    site_id = r["site_id"]
    
    # Skip attempts with no hatch date
    if pd.isna(hatch):
        continue
    
    # Define the 5-day window: hatch-2, hatch-1, hatch, hatch+1, hatch+2
    offsets = [-2, -1, 0, 1, 2]
    dates = [hatch + pd.Timedelta(days=d) for d in offsets]
    
    # For each of the 5 dates, get the mean tmax_F at that site_id (usually 1 row per date)
    temp_map = {}
    for d in dates:
        vals = weather[(weather["site_id"] == site_id) & (weather["date"] == d)]["tmax_F"]
        temp_map[d] = vals.mean() if len(vals) > 0 else np.nan
    
    # Compute averages:
    # pre3 = (hatch-2, hatch-1, hatch)
    pre_vals = [temp_map[dates[0]], temp_map[dates[1]], temp_map[dates[2]]]
    avg_pre3 = np.nanmean(pre_vals) if not all(np.isnan(pre_vals)) else np.nan
    
    # post3 = (hatch, hatch+1, hatch+2)
    post_vals = [temp_map[dates[2]], temp_map[dates[3]], temp_map[dates[4]]]
    avg_post3 = np.nanmean(post_vals) if not all(np.isnan(post_vals)) else np.nan
    
    # 5-day window = hatch-2 .. hatch+2
    all_vals = list(temp_map.values())
    avg_5day = np.nanmean(all_vals) if not all(np.isnan(all_vals)) else np.nan
    
    # OPTIONAL FILTER:
    # If you want to drop rows where we have no temperature info at all, you can check:
    # if np.isnan(avg_5day):
    #     continue
    
    rows.append({
        "attempt_id": r["attempt_id"],
        "site_id": site_id,
        "hatch_date": hatch.strftime("%Y-%m-%d"),
        "avg_pre3": avg_pre3,
        "avg_post3": avg_post3,
        "avg_5day": avg_5day,
    })

# Build DataFrame and save
df = pd.DataFrame(rows)
df.to_csv(OUTPUT_PATH, index=False)

print(f"Saved {len(df)} rows to {OUTPUT_PATH}")
print(df.head())
