import xarray as xr
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

# 1. 讀取數據
heatwave_events = xr.open_dataset(
    'heatwave_events_95threshold_5deg(1965-2024)_k3.nc'
)['t']

tau_max = 10
events_flat = heatwave_events.stack(grid=['latitude', 'longitude']).transpose('valid_time', 'grid')
n_time, n_grid = events_flat.shape

dates = heatwave_events['valid_time'].values
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
def process_pair(i, j):
    Q, es_ij, pairs = compute_es(events_flat[:, i].values,
                                 events_flat[:, j].values,
                                 tau_max)
    return i, j, Q, es_ij, pairs

# 4. 並行計算
results = Parallel(n_jobs=-1, backend="loky", verbose=5)(
    delayed(process_pair)(i, j) for i in range(n_grid) for j in range(i + 1, n_grid)
)

# 5. 生成事件單位
from collections import defaultdict

event_dict = defaultdict(set)  # key: time index, value: set of node indices

for i, j, Q, es_ij, pairs in results:
    for t1, t2 in pairs:
        event_dict[t1].add(i)
        event_dict[t1].add(j)
        event_dict[t2].add(i)
        event_dict[t2].add(j)

# 6. 將相鄰時間的格點合併為單個事件
events_list = []
used_times = set()
event_id = 0

for time_idx in sorted(event_dict.keys()):
    if time_idx in used_times:
        continue

    # 初始化事件
    current_nodes = set(event_dict[time_idx])
    current_times = [time_idx]

    # 往後合併連續時間索引
    next_idx = time_idx + 1
    while next_idx in event_dict and len(event_dict[next_idx] & current_nodes) > 0:
        current_nodes.update(event_dict[next_idx])
        current_times.append(next_idx)
        used_times.add(next_idx)
        next_idx += 1

    # 儲存事件資訊
    events_list.append({
        'event_id': event_id,
        'nodes': list(current_nodes),
        'times': [str(pd.to_datetime(dates[t])) for t in current_times],
        'latitudes': [lat[n] for n in current_nodes],
        'longitudes': [lon[n] for n in current_nodes]
    })
    event_id += 1

# 7. 儲存 CSV
events_df = pd.DataFrame(events_list)
events_df.to_csv('events_transactions_95threshold.csv', index=False)

print("✅ 事件單位 CSV 生成完成")