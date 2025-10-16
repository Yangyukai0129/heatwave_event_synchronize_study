import pandas as pd
import re
import ast

# -----------------------------
# 1. 讀取資料
# -----------------------------
df_events = pd.read_csv('events_transactions_95threshold_post30y.csv')
df_clusters = pd.read_csv('Hierarchical clustering/hierarchical_clustering_result_95threshold_post30y.csv')

# -----------------------------
# 2. 將 nodes / latitudes / longitudes 轉成 list
# -----------------------------
df_events['nodes'] = df_events['nodes'].apply(ast.literal_eval)

def parse_np_list(s):
    nums = re.findall(r"[-+]?\d*\.\d+|\d+", s)
    return [float(x) for x in nums]

df_events['latitudes'] = df_events['latitudes'].apply(parse_np_list)
df_events['longitudes'] = df_events['longitudes'].apply(parse_np_list)

# -----------------------------
# 3. 生成 lat_lon 字串
# -----------------------------
df_events['lat_lon'] = df_events.apply(
    lambda row: [f"{lat}_{lon}" for lat, lon in zip(row['latitudes'], row['longitudes'])],
    axis=1
)

# -----------------------------
# 4. 對齊 list 長度，避免 explode 出錯
# -----------------------------
def align_lists(row):
    min_len = min(len(row['nodes']), len(row['lat_lon']))
    row['nodes'] = row['nodes'][:min_len]
    row['lat_lon'] = row['lat_lon'][:min_len]
    return row

df_events = df_events.apply(align_lists, axis=1)

# -----------------------------
# 5. explode nodes 和 lat_lon
# -----------------------------
df_exploded = df_events.explode(['nodes','lat_lon'])

# -----------------------------
# 6. 合併 cluster
# -----------------------------
df_exploded = df_exploded.merge(df_clusters, left_on='lat_lon', right_on='node', how='left')


# -----------------------------
# 7. 篩選特定群數 (例如 cluster 9)
# -----------------------------
k =[12, 16, 17, 22, 42, 41, 36, 27, 26, 60, 61, 68, 71, 72, 45, 50, 59, 76, 74, 91, 87, 86, 81, 80, 79, 115, 120, 128, 132, 136, 140, 92, 95, 96, 99, 101, 102, 105, 106, 107, 113, 146, 150, 159, 162, 163, 185, 142, 143, 186, 21, 18, 25, 19, 15, 193, 78]
# k = [6,9,11]
cluster_k_events = df_exploded[df_exploded['cluster'].isin(k)]

# -----------------------------
# 8. 生成 FP-Growth transaction list
# -----------------------------
transactions = cluster_k_events.groupby('event_id')['node'].apply(list).tolist()

# -----------------------------
# 9. 存成csv
# -----------------------------
# 轉成 DataFrame，每行一個事件的 node list（轉成字串）
df_trans = pd.DataFrame({'transaction': [str(t) for t in transactions]})
df_trans.to_csv('Hierarchical clustering/transactions_cluster_95threshold_post30y.csv', index=False)