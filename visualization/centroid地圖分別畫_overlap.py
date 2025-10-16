import pandas as pd
import numpy as np

# -----------------------
# 讀取資料
# -----------------------
frequent_itemsets_df = pd.read_csv(
    'D:\\es\\visualization\\frequent_itemsets_with_overlap_pre30y.csv'
)
clusters_df = pd.read_csv(
    'D:\\es\\consensus_clustering\\community_assignment_95threshold_pre30y.csv'
)

# -----------------------
# 節點解析函數
# -----------------------
def parse_nodes(nodes_str):
    """把 '[lat_lon, lat_lon, ...]' 字串轉成 list"""
    if isinstance(nodes_str, str):
        nodes = nodes_str.strip("[]").replace("'", "").split(", ")
        return [n.strip() for n in nodes if n.strip()]
    return []

def latlon_from_nodes(nodes):
    lats, lons = [], []
    for n in nodes:
        if "_" in n:
            lat_str, lon_str = n.split("_")
            lats.append(float(lat_str))
            lons.append(float(lon_str))
    return lats, lons

def compute_centroid(nodes):
    lats, lons = latlon_from_nodes(nodes)
    if len(lats) == 0:
        return None
    return np.mean(lats), np.mean(lons)

# -----------------------
# 解析 cluster 節點 & centroid
# -----------------------
clusters = clusters_df.groupby('community')['latlon'].apply(list).reset_index()
clusters.rename(columns={'latlon':'nodes','community':'cluster_id'}, inplace=True)

clusters['nodes_parsed'] = clusters['nodes'].apply(lambda lst: [parse_nodes(x) for x in lst])
# flatten 每個 cluster 的 node list
clusters['nodes_parsed'] = clusters['nodes_parsed'].apply(lambda x: [n for sub in x for n in sub])
clusters['centroid'] = clusters['nodes_parsed'].apply(compute_centroid)

# -----------------------
# 只取 overlap=1 的 itemsets
# -----------------------
itemsets_full_overlap = frequent_itemsets_df[
    (frequent_itemsets_df['best_overlap'] == 1.0)
].copy()
itemsets_full_overlap['nodes_parsed'] = itemsets_full_overlap['itemset'].apply(parse_nodes)
itemsets_full_overlap['centroid'] = itemsets_full_overlap['nodes_parsed'].apply(compute_centroid)

# -----------------------
# 找最佳對應 cluster
# -----------------------
def find_best_overlap(itemset_nodes, clusters):
    best_ratio = 0
    best_cluster_id = None
    itemset_set = set(itemset_nodes)
    for _, row in clusters.iterrows():
        cluster_set = set(row['nodes_parsed'])
        if len(itemset_set) == 0:
            ratio = 0
        else:
            ratio = len(itemset_set & cluster_set) / len(itemset_set)
        if ratio > best_ratio:
            best_ratio = ratio
            best_cluster_id = row['cluster_id']  # 回傳 cluster_id
    return best_cluster_id, best_ratio

best_matches = []
for _, row in itemsets_full_overlap.iterrows():
    cluster_id, ratio = find_best_overlap(row['nodes_parsed'], clusters)
    best_matches.append((cluster_id, ratio))

itemsets_full_overlap['best_cluster_idx'] = [x[0] for x in best_matches]
itemsets_full_overlap['best_overlap'] = [x[1] for x in best_matches]

# import os
# import matplotlib.pyplot as plt
# import cartopy.crs as ccrs
# import cartopy.feature as cfeature

# # -----------------------
# # 建立輸出資料夾
# # -----------------------
# output_dir = "D:\\es\\visualization\\clusters_with_itemsets_pre30y"
# os.makedirs(output_dir, exist_ok=True)

# # -----------------------
# # 畫底圖函數
# # -----------------------
# def draw_basemap(ax):
#     ax.add_feature(cfeature.LAND, facecolor='lightgray')
#     ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
#     ax.add_feature(cfeature.COASTLINE)
#     ax.add_feature(cfeature.BORDERS, linestyle=':')
#     ax.set_extent([-180, 180, 0, 90], crs=ccrs.PlateCarree())

# # 對每個 cluster 畫圖
# for cluster_idx, cluster_row in clusters.iterrows():
#     fig = plt.figure(figsize=(12, 8))
#     ax = plt.axes(projection=ccrs.Robinson())
#     draw_basemap(ax)

#     # 畫 cluster 節點
#     cluster_lats, cluster_lons = latlon_from_nodes(cluster_row['nodes_parsed'])
#     ax.plot(cluster_lons, cluster_lats,
#             linestyle='None', marker='o', markersize=4,
#             markerfacecolor='none', markeredgecolor='blue',
#             transform=ccrs.PlateCarree(), label="Cluster nodes")

#     # 畫 cluster centroid
#     if cluster_row['centroid'] is not None:
#         ax.scatter(cluster_row['centroid'][1], cluster_row['centroid'][0],
#                    color='darkblue', s=50, transform=ccrs.PlateCarree(),
#                    label="Cluster centroid")

#     # 找出這個 cluster 對應的所有 overlap=1 itemsets
#     itemsets_for_cluster = itemsets_full_overlap[itemsets_full_overlap['best_cluster_idx'] == cluster_row['cluster_id']]

#     for _, itemset_row in itemsets_for_cluster.iterrows():
#         # 畫 itemset 節點
#         # item_lats, item_lons = latlon_from_nodes(itemset_row['nodes_parsed'])
#         # if len(item_lats) > 0:
#         #     ax.plot(item_lons, item_lats,
#         #             linestyle='None', marker='o', markersize=4,
#         #             markerfacecolor='none', markeredgecolor='red',
#         #             transform=ccrs.PlateCarree(), label="Itemset nodes")

#         # 畫 itemset centroid
#         if itemset_row['centroid'] is not None:
#             ax.scatter(itemset_row['centroid'][1], itemset_row['centroid'][0],
#                        color='darkred', s=40, transform=ccrs.PlateCarree(),
#                        label="Itemset centroid")

#             # 畫線連 cluster centroid
#             c_cluster = cluster_row['centroid']
#             c_itemset = itemset_row['centroid']
#             ax.plot([c_itemset[1], c_cluster[1]],
#                     [c_itemset[0], c_cluster[0]],
#                     color='gray', alpha=0.5, transform=ccrs.PlateCarree())

#     # 避免 legend 重複
#     handles, labels = ax.get_legend_handles_labels()
#     by_label = dict(zip(labels, handles))
#     ax.legend(handles=by_label.values(), labels=by_label.keys(), loc='lower left')

#     ax.set_title(f"Cluster {cluster_row['cluster_id']} and its Itemsets")

#     # 存圖
#     out_path = os.path.join(output_dir, f"cluster_{cluster_row['cluster_id']}.png")
#     plt.savefig(out_path, dpi=150, bbox_inches="tight")
#     plt.close(fig)

# 對每個 itemset 畫圖（每張圖只畫一個 itemset 與其對應 cluster）

# print(f"✅ 已經輸出 {len(clusters)} 張圖到 {output_dir}")

import os
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# -----------------------
# 畫底圖函數
# -----------------------
def draw_basemap(ax):
    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.set_extent([-180, 180, 0, 90], crs=ccrs.PlateCarree())

# -----------------------
# 主繪圖迴圈
# -----------------------
output_dir = "D:\\es\\visualization\\clusters_with_itemsets_pre30y"
os.makedirs(output_dir, exist_ok=True)

# ✅ 設定 itemset 節點最少數
MIN_ELEMENTS = 2

# 新增欄位：每個 itemset 的節點數
itemsets_full_overlap['num_elements'] = itemsets_full_overlap['nodes_parsed'].apply(len)

# 篩選符合條件的 itemset
valid_itemsets = itemsets_full_overlap[itemsets_full_overlap['num_elements'] >= MIN_ELEMENTS]

print(f"📊 僅繪製含有 ≥{MIN_ELEMENTS} 個節點的 itemset，共 {len(valid_itemsets)} 筆。")

for _, itemset_row in valid_itemsets.iterrows():
    cluster_id = itemset_row['best_cluster_idx']
    if cluster_id is None or np.isnan(cluster_id):
        continue  # 跳過沒有對應 cluster 的 itemset

    cluster_row = clusters[clusters['cluster_id'] == cluster_id].iloc[0]

    # 建立子資料夾，例如 D:\es\visualization\clusters_with_itemsets_post30y\cluster_0\
    cluster_folder = os.path.join(output_dir, f"cluster_{cluster_id}")
    os.makedirs(cluster_folder, exist_ok=True)

    fig = plt.figure(figsize=(12, 8))
    ax = plt.axes(projection=ccrs.Robinson())
    draw_basemap(ax)

    # 畫 cluster nodes
    cluster_lats, cluster_lons = latlon_from_nodes(cluster_row['nodes_parsed'])
    ax.plot(cluster_lons, cluster_lats,
            linestyle='None', marker='o', markersize=4,
            markerfacecolor='none', markeredgecolor='blue',
            transform=ccrs.PlateCarree(), label="Cluster nodes")

    # 畫 cluster centroid
    # if cluster_row['centroid'] is not None:
    #     ax.scatter(cluster_row['centroid'][1], cluster_row['centroid'][0],
    #                color='darkblue', s=50, transform=ccrs.PlateCarree(),
    #                label="Cluster centroid")

    # 畫 itemset nodes
    item_lats, item_lons = latlon_from_nodes(itemset_row['nodes_parsed'])
    ax.plot(item_lons, item_lats,
            linestyle='None', marker='.', markersize=4,
            markerfacecolor='none', markeredgecolor='red',
            transform=ccrs.PlateCarree(), label="Itemset nodes")

    # # 畫 itemset centroid
    # if itemset_row['centroid'] is not None:
    #     ax.scatter(itemset_row['centroid'][1], itemset_row['centroid'][0],
    #                color='darkred', s=40, transform=ccrs.PlateCarree(),
    #                label="Itemset centroid")

    #     # 連線
    #     c_cluster = cluster_row['centroid']
    #     c_itemset = itemset_row['centroid']
    #     ax.plot([c_itemset[1], c_cluster[1]],
    #             [c_itemset[0], c_cluster[0]],
    #             color='gray', alpha=0.5, transform=ccrs.PlateCarree())

    # 標題與圖例
    ax.set_title(f"Itemset {itemset_row.name} → Cluster {cluster_id}")
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(handles=by_label.values(), labels=by_label.keys(), loc='lower left')

    # 儲存圖片
    out_path = os.path.join(cluster_folder, f"itemset_{itemset_row.name}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

print("✅ 各 cluster 的 itemset 圖已輸出到：", output_dir)