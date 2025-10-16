import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import ast
import re

plt.rcParams['font.family'] = ['Microsoft JhengHei']  # 或 'SimHei'、'Noto Sans CJK TC'
plt.rcParams['axes.unicode_minus'] = False  # 負號正常顯示

df_clusters = pd.read_csv("Hierarchical clustering/hierarchical_clustering_result_95threshold_post30y.csv")
df_events = pd.read_csv("events_transactions_95threshold_post30y.csv")

# -----------------------------
# 將 nodes / latitudes / longitudes 轉成 list
# -----------------------------
df_events['nodes'] = df_events['nodes'].apply(ast.literal_eval)

def parse_np_list(s):
    nums = re.findall(r"[-+]?\d*\.\d+|\d+", s)
    return [float(x) for x in nums]

df_events['latitudes'] = df_events['latitudes'].apply(parse_np_list)
df_events['longitudes'] = df_events['longitudes'].apply(parse_np_list)

# -----------------------------
# 生成 lat_lon 字串
# -----------------------------
df_events['lat_lon'] = df_events.apply(
    lambda row: [f"{lat}_{lon}" for lat, lon in zip(row['latitudes'], row['longitudes'])],
    axis=1
)

# ===== Step 1️⃣：統計所有格點的出現次數與高頻門檻 =====
from collections import Counter
import numpy as np

all_nodes = [n for nodes in df_events['lat_lon'] for n in nodes]
node_counts = Counter(all_nodes)
node_freq = pd.DataFrame(node_counts.items(), columns=['node', 'count'])
threshold = np.percentile(node_freq['count'], 75)  # 取第 75 百分位數（即前 25% 出現次數最多的格點） 當作門檻值
high_freq_nodes = set(node_freq[node_freq['count'] >= threshold]['node']) # 凡是出現次數 大於或等於該門檻的格點，都被歸為「高頻熱點」

print(f"高頻熱點門檻 = {threshold}，共 {len(high_freq_nodes)} 個高頻格點")

# ===== Step 2️⃣：計算各群中高頻熱點比例 =====
df_clusters['is_highfreq'] = df_clusters['node'].isin(high_freq_nodes)

cluster_stats = (
    df_clusters.groupby('cluster')['is_highfreq']
    .agg(['sum', 'count'])
    .rename(columns={'sum': 'highfreq_nodes', 'count': 'total_nodes'})
)
cluster_stats['highfreq_ratio'] = cluster_stats['highfreq_nodes'] / cluster_stats['total_nodes']

print(cluster_stats.sort_values('highfreq_ratio', ascending=False))

#  ===== Step 3️⃣：視覺化群內高頻熱點比例 =====
import matplotlib.pyplot as plt

plt.figure(figsize=(8,4))
plt.bar(cluster_stats.index, cluster_stats['highfreq_ratio'])
plt.axhline(0.5, color='red', linestyle='--', label='50% 門檻')
plt.xlabel('Cluster ID')
plt.ylabel('高頻熱點比例')
plt.title('各群中高頻熱點佔比(post30y)')
plt.legend()
plt.show()

# ===== Step 4️⃣：篩選出高頻熱點比例超過 50% 的群 =====
highfreq_clusters = cluster_stats[cluster_stats['highfreq_ratio'] > 0.5]

print("\n🔥 高頻熱點比例超過 50% 的群如下：")
print(highfreq_clusters.sort_values('highfreq_ratio', ascending=False).index.tolist())
print(f"\n共有 {len(highfreq_clusters)} 個群符合條件。")