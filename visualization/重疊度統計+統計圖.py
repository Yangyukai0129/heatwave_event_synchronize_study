import pandas as pd
import numpy as np
import ast
from joblib import Parallel, delayed
import seaborn as sns
import matplotlib.pyplot as plt

# # -----------------------
# # 讀取資料
# # -----------------------
# frequent_itemsets_df = pd.read_csv('D:\\es\\frequent_itemset\\frequent_itemsets_95threshold_detrend_post30y.csv')
# clusters_df = pd.read_csv('D:\\es\\consensus_clustering\\community_assignment_95threshold_detrend_post30y.csv')

# # -----------------------
# # 將 cluster_df 轉成 cluster_id -> nodes list
# # -----------------------
# clusters = clusters_df.groupby('community')['latlon'].apply(list).reset_index()
# clusters.rename(columns={'latlon':'nodes','community':'cluster_id'}, inplace=True)

# # -----------------------
# # 節點解析與 centroid 計算
# # -----------------------
# def parse_nodes(nodes_str):
#     if isinstance(nodes_str, str):
#         nodes = nodes_str.strip("[]").replace("'", "").split(", ")
#         return nodes
#     return []

# def latlon_from_nodes(nodes):
#     lats, lons = [], []
#     for n in nodes:
#         if "_" in n:
#             lat_str, lon_str = n.split("_")
#             lats.append(float(lat_str))
#             lons.append(float(lon_str))
#     return lats, lons

# def compute_centroid(nodes):
#     lats, lons = latlon_from_nodes(nodes)
#     if len(lats) == 0:
#         return None
#     return np.mean(lats), np.mean(lons)

# # frequent itemsets 解析
# frequent_itemsets_df['nodes_parsed'] = frequent_itemsets_df['itemset'].apply(parse_nodes)
# frequent_itemsets_df['centroid'] = frequent_itemsets_df['nodes_parsed'].apply(compute_centroid)

# # clusters 解析
# clusters['nodes_parsed'] = clusters['nodes'].apply(lambda x: [n.strip() for n in x])
# clusters['centroid'] = clusters['nodes_parsed'].apply(compute_centroid)

# # -----------------------
# # 計算 best_overlap（並行化）
# # -----------------------
# def find_best_overlap(itemset_nodes, clusters):
#     best_ratio = 0
#     for _, row in clusters.iterrows():
#         cluster_set = set(row['nodes_parsed'])
#         itemset_set = set(itemset_nodes)
#         if len(itemset_set) > 0:
#             ratio = len(itemset_set & cluster_set) / len(itemset_set)
#         else:
#             ratio = 0
#         if ratio > best_ratio:
#             best_ratio = ratio
#     return best_ratio

# # 並行化計算
# frequent_itemsets_df['best_overlap'] = Parallel(n_jobs=-1, verbose=5)(
#     delayed(find_best_overlap)(x, clusters) for x in frequent_itemsets_df['nodes_parsed']
# )

# # -----------------------
# # 儲存 CSV
# # -----------------------
# frequent_itemsets_df.to_csv('visualization//frequent_itemsets_with_overlap_detrend_post30y.csv', index=False)
# print("✅ 已計算 best_overlap 並儲存 CSV")

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# -----------------------
# 讀取 CSV
# -----------------------
pre_df = pd.read_csv('D:\\es\\visualization\\frequent_itemsets_with_overlap_detrend_pre30y.csv')
post_df = pd.read_csv('D:\\es\\visualization\\frequent_itemsets_with_overlap_detrend_post30y.csv')

# -----------------------
# 建立繪圖
# -----------------------
# 加入時段標籤
pre_df['period'] = 'Pre 30 Years (1965-1994)'
post_df['period'] = 'Post 30 Years (1995-2024)'

# 合併資料
combined_df = pd.concat([pre_df, post_df], ignore_index=True)

# -----------------------
# 1️⃣ 小提琴圖
# -----------------------
plt.figure(figsize=(10,6))
sns.violinplot(x='period', y='best_overlap', data=combined_df, inner='quartile')
plt.ylabel('Overlap Ratio')
plt.title('Violin Plot: Frequent Itemset vs Cluster Overlap')
plt.show()

# -----------------------
# 2️⃣ 箱型圖
# -----------------------
plt.figure(figsize=(10,6))
sns.boxplot(x='period', y='best_overlap', data=combined_df)
plt.ylabel('Overlap Ratio')
plt.title('Boxplot: Frequent Itemset vs Cluster Overlap')
plt.show()