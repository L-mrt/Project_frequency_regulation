import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Paths
DATA_PATH = 'data/driving_sessions.csv'
OUT_DIR = 'images'

if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)

# Load driving sessions
# File uses ';' separator and has START/STOP columns
df = pd.read_csv(DATA_PATH, sep=';')

# Parse datetimes
for col in ['START', 'STOP']:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], utc=True, errors='coerce')
        df[col] = df[col].dt.tz_convert(None)

# Drop rows with missing START/STOP for now (SOC_STOP handled below)
df = df.dropna(subset=['START', 'STOP'])

# Ensure necessary SOC columns exist
if 'SOC_START' not in df.columns or 'SOC_STOP' not in df.columns:
    raise SystemExit('SOC_START and SOC_STOP columns required in driving_sessions.csv')

# Drop rows with missing START/STOP/SOC
df = df.dropna(subset=['START', 'STOP', 'SOC_START', 'SOC_STOP'])

# Clip SOC to [0,100] in case of bad data
df['SOC_START'] = df['SOC_START'].clip(0, 100)
df['SOC_STOP'] = df['SOC_STOP'].clip(0, 100)

# Battery capacity: 46 kWh (project setting)
BAT_CAP = 46.0
CHARGE_POWER_KW = 7.0

# Sort by vehicle ID and trip start time to compute parking sessions
if 'ID' in df.columns:
    df = df.sort_values(['ID', 'START']).reset_index(drop=True)
    # Get next trip start for same vehicle
    df['NEXT_START'] = df.groupby('ID')['START'].shift(-1)
else:
    # If no ID column, assume single vehicle
    df = df.sort_values('START').reset_index(drop=True)
    df['NEXT_START'] = df['START'].shift(-1)

# Parking duration = time between end of trip N and start of trip N+1 (in hours)
df['PARKING_DURATION_H'] = (df['NEXT_START'] - df['STOP']).dt.total_seconds() / 3600.0

# Energy consumed during trip N (in kWh)
df['ENERGY_CONSUMED_kWh'] = ((df['SOC_START'] - df['SOC_STOP']) / 100.0) * BAT_CAP

# Max energy recoverable during parking at 7 kW
df['E_MAX_AC_kWh'] = CHARGE_POWER_KW * df['PARKING_DURATION_H']

# AC if parking duration allows recovering the consumed energy at 7 kW
# DC otherwise (user had to use fast charger)
# Cast to object to allow NaN for last trips
df['IS_AC'] = (df['E_MAX_AC_kWh'] >= df['ENERGY_CONSUMED_kWh']) & (df['PARKING_DURATION_H'] > 0)
df['IS_AC'] = df['IS_AC'].astype(object)

# For last trip of each vehicle (no next start), mark as unknown/excluded
df.loc[df['NEXT_START'].isna(), 'IS_AC'] = np.nan

# Drop rows without parking info (last trip per vehicle)
df = df.dropna(subset=['IS_AC'])

# Build hourly time index from min STOP to max NEXT_START (parking periods)
start = df['STOP'].min().floor('h')
end = df['NEXT_START'].max().ceil('h')
index = pd.date_range(start=start, end=end, freq='h')
counts_ac = pd.Series(0, index=index, dtype=int)
counts_dc = pd.Series(0, index=index, dtype=int)

# For each session, add 1 to hourly bins between STOP (start of parking) and NEXT_START (end of parking)
for _, row in df.iterrows():
    # Parking period: from STOP to NEXT_START
    s = row['STOP'].ceil('h') if row['STOP'].minute != 0 or row['STOP'].second != 0 else row['STOP']
    e = row['NEXT_START'].floor('h')
    if e < s:
        # If shorter than an hour, still add to the hour containing STOP
        hr = pd.DatetimeIndex([row['STOP'].floor('h')])
    else:
        hr = pd.date_range(start=s, end=e, freq='h')
    if len(hr) == 0:
        hr = pd.DatetimeIndex([row['STOP'].floor('h')])

    if row['IS_AC']:
        counts_ac.loc[counts_ac.index.isin(hr)] += 1
    else:
        counts_dc.loc[counts_dc.index.isin(hr)] += 1

# Plot
plt.figure(figsize=(12, 5))
plt.plot(counts_ac.index, counts_ac.values, label='AC (slow, eligible FCR)', color='tab:green')
plt.plot(counts_dc.index, counts_dc.values, label='DC (fast, not eligible)', color='tab:red')
plt.fill_between(counts_ac.index, counts_ac.values, alpha=0.15, color='tab:green')
plt.fill_between(counts_dc.index, counts_dc.values, alpha=0.15, color='tab:red')
plt.xlabel('Time')
plt.ylabel('Number of vehicles charging')
plt.title('AC vs DC charging sessions (hourly counts)')
plt.legend()
plt.grid(True, alpha=0.3)

# On-plot summary: totals and percentages (no console print)
total_ac = counts_ac.sum()
total_dc = counts_dc.sum()
total = total_ac + total_dc
if total > 0:
    pct_ac = 100.0 * total_ac / total
    pct_dc = 100.0 * total_dc / total
else:
    pct_ac = pct_dc = 0.0

summary_text = f"AC={int(total_ac)} ({pct_ac:.1f}%), DC={int(total_dc)} ({pct_dc:.1f}%) | Logic: AC if 7kW*T_park >= Energy_consumed"
plt.annotate(summary_text, xy=(0.01, 0.98), xycoords='axes fraction', fontsize=9, va='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.show()
