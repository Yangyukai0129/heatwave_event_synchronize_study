import xarray as xr 
import numpy as np 
import pandas as pd 
from joblib import Parallel, delayed 
from collections import defaultdict 
# 1. 讀取數據 
heatwave_events = xr.open_dataset('data/heatwave_event/heatwave_events_95threshold_5deg(1965-2024)_k3.nc')['t'] 
tau_max = 10 # 將資料壓平 
events_flat = heatwave_events.stack(grid=['latitude', 'longitude']).transpose('valid_time', 'grid') 
n_time, n_grid = events_flat.shape 
dates = events_flat['valid_time'].values 
lat = events_flat['latitude'].values 
lon = events_flat['longitude'].values 
# 2. ES 函數 
def compute_es(events_i, events_j, tau_max):
    t_i = np.where(events_i == 1)[0]
    t_j = np.where(events_j == 1)[0]
    n_i, n_j = len(t_i), len(t_j)

    if n_i == 0 or n_j == 0:
        return 0.0, 0, np.empty((0, 2), dtype=np.int32)

    intervals_i = np.diff(t_i) if n_i > 1 else np.array([])
    intervals_j = np.diff(t_j) if n_j > 1 else np.array([])

    tau_i = np.minimum(
        np.concatenate(([np.inf], intervals_i)),
        np.concatenate((intervals_i, [np.inf]))
    )
    tau_j = np.minimum(
        np.concatenate(([np.inf], intervals_j)),
        np.concatenate((intervals_j, [np.inf]))
    )

    es_ij = 0
    pairs = []
    c_ij = 0.0
    for a in range(n_i):
        for b in range(n_j):
            t_ij = abs(t_i[a] - t_j[b])
            tau_ab = 0.5 * min(tau_i[a], tau_j[b])

            if t_ij < tau_ab and t_ij <= tau_max:
                es_ij += 1
                pairs.append((t_i[a], t_j[b]))

            if 0 < t_ij <= tau_max:
                c_ij += 0.5 * min(t_ij, tau_max)
    Q = c_ij / np.sqrt(n_i * n_j) if n_i * n_j > 0 else 0.0
    return min(Q, 1.0), es_ij, np.array(pairs, dtype=np.int32)

# 3. 包裝函數 for joblib 
def process_pair(i, j, events_subset): 
    Q, es_ij, pairs = compute_es(events_subset[:, i].values, events_subset[:, j].values, tau_max)
    return i, j, Q, es_ij, pairs 
# 4. 事件生成函數 
def generate_events_csv(events_subset, dates_subset, filename):

    n_time, n_grid = events_subset.shape
    grid_index = events_subset['grid'].to_index()
    lat = grid_index.get_level_values('latitude').values
    lon = grid_index.get_level_values('longitude').values

    # 🔥 1. 計算所有 pair 的同步事件
    results = Parallel(n_jobs=-1, backend="loky", verbose=5)(
        delayed(process_pair)(i, j, events_subset)
        for i in range(n_grid)
        for j in range(i + 1, n_grid)
    )

    # 🔥 2. 將所有 pairs 放進 event_dict
    event_dict = defaultdict(set)

    for i, j, Q, es_ij, pairs in results:
        for t1, t2 in pairs:
            event_dict[t1].add(i)
            event_dict[t1].add(j)
            event_dict[t2].add(i)
            event_dict[t2].add(j)

    # 🔥 3. 將 event_dict 轉成 event_list（事件串接）
    events_list = []
    used_times = set()
    event_id = 0

    for time_idx in sorted(event_dict.keys()):
        if time_idx in used_times:
            continue

        current_nodes = set(event_dict[time_idx])
        current_times = [time_idx]

        next_idx = time_idx + 1

        while next_idx in event_dict and len(event_dict[next_idx] & current_nodes) > 0:
            current_nodes.update(event_dict[next_idx])
            current_times.append(next_idx)
            used_times.add(next_idx)
            next_idx += 1

        events_list.append({
            'event_id': event_id,
            'nodes': list(current_nodes),
            'times': [str(pd.to_datetime(dates_subset[t])) for t in current_times],
            'latitudes': [lat[n] for n in current_nodes],
            'longitudes': [lon[n] for n in current_nodes]
        })

        event_id += 1

    # 🔥 4. 輸出 CSV
    events_df = pd.DataFrame(events_list)
    events_df.to_csv(filename, index=False)
    print(f"✅ 事件單位 CSV 生成完成: {filename}")
# 5. 切分前後 30 年 
events_flat_pre = events_flat.sel(valid_time=slice("1965-06-01", "1994-08-31")) 
dates_pre = events_flat_pre['valid_time'].values 
events_flat_post = events_flat.sel(valid_time=slice("1995-06-01", "2024-08-31")) 
dates_post = events_flat_post['valid_time'].values 
# 6. 生成前後期 CSV 
generate_events_csv(events_flat_pre, dates_pre, 'events_transactions_95threshold_detrend_pre30y.csv') 
generate_events_csv(events_flat_post, dates_post, 'events_transactions_95threshold_detrend_post30y.csv')