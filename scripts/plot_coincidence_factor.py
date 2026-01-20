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
df = pd.read_csv(DATA_PATH, sep=';')

# Parse datetimes
for col in ['START', 'STOP']:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], utc=True, errors='coerce')
        df[col] = df[col].dt.tz_convert(None)

# Drop rows with missing START/STOP
df = df.dropna(subset=['START', 'STOP'])

# Ensure necessary SOC columns exist
if 'SOC_START' not in df.columns or 'SOC_STOP' not in df.columns:
    raise SystemExit('SOC_START and SOC_STOP columns required in driving_sessions.csv')

# Drop rows with missing SOC
df = df.dropna(subset=['SOC_START', 'SOC_STOP'])

# Clip SOC to [0,100]
df['SOC_START'] = df['SOC_START'].clip(0, 100)
df['SOC_STOP'] = df['SOC_STOP'].clip(0, 100)

# Battery capacity: 46 kWh
BAT_CAP = 46.0
CHARGE_POWER_KW = 7.0

# Sort by vehicle ID and trip start time
if 'ID' in df.columns:
    df = df.sort_values(['ID', 'START']).reset_index(drop=True)
    df['NEXT_START'] = df.groupby('ID')['START'].shift(-1)
else:
    df = df.sort_values('START').reset_index(drop=True)
    df['NEXT_START'] = df['START'].shift(-1)

# Parking duration = time between end of trip N and start of trip N+1 (in hours)
df['PARKING_DURATION_H'] = (df['NEXT_START'] - df['STOP']).dt.total_seconds() / 3600.0

# Energy consumed during trip N (in kWh)
df['ENERGY_CONSUMED_kWh'] = ((df['SOC_START'] - df['SOC_STOP']) / 100.0) * BAT_CAP

# Max energy recoverable during parking at 7 kW
df['E_MAX_AC_kWh'] = CHARGE_POWER_KW * df['PARKING_DURATION_H']

# AC if parking duration allows recovering the consumed energy at 7 kW
df['IS_AC'] = (df['E_MAX_AC_kWh'] >= df['ENERGY_CONSUMED_kWh']) & (df['PARKING_DURATION_H'] > 0)
df['IS_AC'] = df['IS_AC'].astype(object)
df.loc[df['NEXT_START'].isna(), 'IS_AC'] = np.nan

# Drop rows without parking info
df = df.dropna(subset=['IS_AC'])
df['IS_AC'] = df['IS_AC'].astype(bool)

# Filter to keep only AC sessions (eligible for V2G)
df_ac = df[df['IS_AC']].copy()

print(f"Total sessions: {len(df)}")
print(f"AC sessions: {len(df_ac)} ({100*len(df_ac)/len(df):.1f}%)")
print(f"DC sessions: {len(df)-len(df_ac)} ({100*(len(df)-len(df_ac))/len(df):.1f}%)")

# Create time index at 10-minute resolution for vectorized computation
time_start = df_ac['STOP'].min().floor('h')
time_end = df_ac['NEXT_START'].max().ceil('h')
time_index = pd.date_range(start=time_start, end=time_end, freq='10min')

print(f"\nTime range: {time_start} to {time_end}")
print(f"Time steps (10min): {len(time_index)}")

# Vectorized approach: create boolean matrix (time_steps x sessions)
# Each column = one AC session, each row = one 10-min timestep
# Value = True if session is active (vehicle plugged in AC) at that timestep

n_steps = len(time_index)
n_sessions = len(df_ac)

# Preallocate boolean matrix
active_matrix = np.zeros((n_steps, n_sessions), dtype=bool)

print(f"Building occupancy matrix ({n_steps} x {n_sessions})...")

# Convert session start/end to indices
df_ac['stop_idx'] = df_ac['STOP'].apply(lambda x: time_index.searchsorted(x))
df_ac['next_start_idx'] = df_ac['NEXT_START'].apply(lambda x: time_index.searchsorted(x))

# Vectorized fill: mark active periods
for i, row in enumerate(df_ac.itertuples()):
    start_idx = row.stop_idx
    end_idx = row.next_start_idx
    if start_idx < n_steps and end_idx > start_idx:
        end_idx = min(end_idx, n_steps)
        active_matrix[start_idx:end_idx, i] = True

# Count simultaneous vehicles at each timestep
n_simultaneous = active_matrix.sum(axis=1)

# Create DataFrame for easier manipulation
availability_df = pd.DataFrame({
    'time': time_index,
    'n_vehicles': n_simultaneous
})

print(f"Max simultaneous AC vehicles: {n_simultaneous.max()}")
print(f"Mean simultaneous AC vehicles: {n_simultaneous.mean():.1f}")

# Calculate coincidence factor for 1h and 4h blocks
# Coincidence factor = minimum number of EVs simultaneously plugged in AC over the period

# 1-hour blocks (12pm-1am, 1am-2am, etc.)
availability_df['hour_block'] = availability_df['time'].dt.floor('h')
cf_1h = availability_df.groupby('hour_block')['n_vehicles'].min()

# 4-hour blocks (12pm-4am, 4am-8am, etc.)
# Create 4-hour block labels
def get_4h_block(dt):
    hour = dt.hour
    block_start = (hour // 4) * 4
    return dt.replace(hour=block_start, minute=0, second=0, microsecond=0)

availability_df['block_4h'] = availability_df['time'].apply(get_4h_block)
cf_4h = availability_df.groupby('block_4h')['n_vehicles'].min()

print(f"\n1h blocks: {len(cf_1h)} blocks")
print(f"4h blocks: {len(cf_4h)} blocks")

# Plot coincidence factor over time
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=False)

# 1-hour coincidence factor
ax1.plot(cf_1h.index, cf_1h.values, marker='o', markersize=2, linewidth=0.5, label='1h blocks', color='tab:blue')
ax1.fill_between(cf_1h.index, cf_1h.values, alpha=0.2, color='tab:blue')
ax1.set_ylabel('Coincidence Factor\n(min simultaneous AC vehicles)', fontsize=10)
ax1.set_title('Coincidence Factor for 1-hour blocks', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend()

# Stats annotation
mean_1h = cf_1h.mean()
min_1h = cf_1h.min()
max_1h = cf_1h.max()
stats_text_1h = f"Mean={mean_1h:.1f} | Min={min_1h} | Max={max_1h}"
ax1.annotate(stats_text_1h, xy=(0.01, 0.98), xycoords='axes fraction', fontsize=9, va='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# 4-hour coincidence factor
ax2.plot(cf_4h.index, cf_4h.values, marker='s', markersize=3, linewidth=1.0, label='4h blocks', color='tab:orange')
ax2.fill_between(cf_4h.index, cf_4h.values, alpha=0.2, color='tab:orange')
ax2.set_xlabel('Time', fontsize=10)
ax2.set_ylabel('Coincidence Factor\n(min simultaneous AC vehicles)', fontsize=10)
ax2.set_title('Coincidence Factor for 4-hour blocks', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend()

# Stats annotation
mean_4h = cf_4h.mean()
min_4h = cf_4h.min()
max_4h = cf_4h.max()
stats_text_4h = f"Mean={mean_4h:.1f} | Min={min_4h} | Max={max_4h}"
ax2.annotate(stats_text_4h, xy=(0.01, 0.98), xycoords='axes fraction', fontsize=9, va='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.show()

# Save data for later use (optional)
cf_1h.to_csv('data/availability_1h.csv', header=['min_vehicles'])
cf_4h.to_csv('data/availability_4h.csv', header=['min_vehicles'])
print(f"\nSaved coincidence factor data to data/availability_1h.csv and data/availability_4h.csv")
