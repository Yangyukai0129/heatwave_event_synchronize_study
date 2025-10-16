import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# -----------------------
# 讀取資料
# -----------------------
frequent_itemsets_df = pd.read_csv('D:\\es\\frequent_itemset\\frequent_itemsets_95threshold_post30y.csv')
frequent_itemsets_df = frequent_itemsets_df.sample(300, random_state=42)
clusters_df = pd.read_csv('D:\\es\\consensus_clustering\\community_assignment_95threshold_post30y.csv')

# 將 cluster_df 轉成 cluster_id -> nodes list
clusters = clusters_df.groupby('community')['latlon'].apply(list).reset_index()
clusters.rename(columns={'latlon':'nodes','community':'cluster_id'}, inplace=True)

# -----------------------
# 解析節點
# -----------------------
def parse_nodes(nodes_str):
    if isinstance(nodes_str, str):
        nodes = nodes_str.strip("[]").replace("'", "").split(", ")
        return nodes
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
# 找每個 itemset 對應最重疊 cluster
# -----------------------
def find_best_overlap(itemset_nodes, clusters):
    best_ratio = 0
    best_cluster_idx = None
    itemset_set = set(itemset_nodes)
    for idx, row in clusters.iterrows():
        cluster_set = set(row['nodes_parsed'])
        if len(itemset_set) == 0:
            ratio = 0
        else:
            ratio = len(itemset_set & cluster_set) / len(itemset_set)
        if ratio > best_ratio:
            best_ratio = ratio
            best_cluster_idx = idx
    return best_cluster_idx, best_ratio

# 解析節點 & centroid
frequent_itemsets_df['nodes_parsed'] = frequent_itemsets_df['itemset'].apply(parse_nodes)
frequent_itemsets_df['centroid'] = frequent_itemsets_df['nodes_parsed'].apply(compute_centroid)

# 先處理 clusters 的 nodes_parsed & centroid
clusters['nodes_parsed'] = clusters['nodes'].apply(lambda x: [n.strip() for n in x])
clusters['centroid'] = clusters['nodes_parsed'].apply(compute_centroid)

# 再去找最佳對應
best_matches = []
for idx, row in frequent_itemsets_df.iterrows():
    best_idx, best_ratio = find_best_overlap(row['nodes_parsed'], clusters)
    best_matches.append((best_idx, best_ratio))

frequent_itemsets_df['best_cluster_idx'] = [x[0] for x in best_matches]
frequent_itemsets_df['best_overlap'] = [x[1] for x in best_matches]

# # 每個 cluster 畫一張圖
# for cluster_idx, cluster_row in clusters.iterrows():
#     fig = plt.figure(figsize=(12, 8))
#     ax = plt.axes(projection=ccrs.Robinson())
#     ax.add_feature(cfeature.LAND, facecolor='lightgray')
#     ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
#     ax.add_feature(cfeature.COASTLINE)
#     ax.add_feature(cfeature.BORDERS, linestyle=':')
#     ax.set_extent([-180, 180, 0, 90], crs=ccrs.PlateCarree())

#     # 畫 cluster 節點
#     lats, lons = latlon_from_nodes(cluster_row['nodes_parsed'])
#     ax.plot(lons, lats,
#         linestyle='None', marker='o', markersize=4,
#         markerfacecolor='none', markeredgecolor='blue',
#         transform=ccrs.PlateCarree(), label="Cluster nodes")

#     # 畫 cluster centroid
#     if cluster_row['centroid'] is not None:
#         ax.scatter(cluster_row['centroid'][1], cluster_row['centroid'][0],
#                    color='darkblue', s=50, transform=ccrs.PlateCarree(),
#                    label="Cluster centroid")

#     # 找對應到這個 cluster 的 itemsets
#     itemsets_for_cluster = frequent_itemsets_df[frequent_itemsets_df['best_cluster_idx'] == cluster_idx]

#     for _, itemset_row in itemsets_for_cluster.iterrows():
#         if itemset_row['centroid'] is not None:
#             # 畫 itemset 所有點
#             # item_lats, item_lons = latlon_from_nodes(itemset_row['nodes_parsed'])
#             # ax.scatter(item_lons, item_lats,
#             #            color='orange', alpha=0.6, s=15,
#             #            transform=ccrs.PlateCarree(), label="Itemset nodes")

#             # 畫 itemset centroid
#             ax.scatter(itemset_row['centroid'][1], itemset_row['centroid'][0],
#                        color='darkred', s=40, transform=ccrs.PlateCarree(),
#                        label="Itemset centroid")

#             # 畫線連結
#             c_cluster = cluster_row['centroid']
#             c_itemset = itemset_row['centroid']
#             ax.plot([c_itemset[1], c_cluster[1]], [c_itemset[0], c_cluster[0]],
#                     color='gray', alpha=0.5, transform=ccrs.PlateCarree())

#     # 避免 legend 重複
#     handles, labels = ax.get_legend_handles_labels()
#     by_label = dict(zip(labels, handles))
#     ax.legend(by_label.values(), by_label.keys())

#     plt.title(f"Cluster {cluster_idx} and its Itemsets \n Pre 30 Years (1965-1994)")
#     plt.show()
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# -----------------------
# 畫底圖
# -----------------------
def draw_basemap(ax):
    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.set_extent([-180, 180, 0, 90], crs=ccrs.PlateCarree())

# -----------------------
# 更新函數 (每一幀)
# -----------------------
def update(frame_idx):
    ax.clear()
    draw_basemap(ax)

    cluster_row = clusters.iloc[frame_idx]

    # 畫 cluster 節點
    lats, lons = latlon_from_nodes(cluster_row['nodes_parsed'])
    ax.plot(lons, lats,
            linestyle='None', marker='o', markersize=4,
            markerfacecolor='none', markeredgecolor='blue',
            transform=ccrs.PlateCarree(), label="Cluster nodes")

    # 畫 cluster centroid
    if cluster_row['centroid'] is not None:
        ax.scatter(cluster_row['centroid'][1], cluster_row['centroid'][0],
                   color='darkblue', s=50, transform=ccrs.PlateCarree(),
                   label="Cluster centroid")

    # 找對應的 itemsets
    itemsets_for_cluster = frequent_itemsets_df[
        frequent_itemsets_df['best_cluster_idx'] == frame_idx
    ]


    for _, itemset_row in itemsets_for_cluster.iterrows():
        if itemset_row['centroid'] is not None:
            ax.scatter(itemset_row['centroid'][1], itemset_row['centroid'][0],
                       color='darkred', s=40, transform=ccrs.PlateCarree(),
                       label="Itemset centroid")

            # 畫線連到 cluster centroid
            c_cluster = cluster_row['centroid']
            c_itemset = itemset_row['centroid']
            ax.plot([c_itemset[1], c_cluster[1]],
                    [c_itemset[0], c_cluster[0]],
                    color='gray', alpha=0.5, transform=ccrs.PlateCarree())

    # 避免 legend 重複
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(handles=by_label.values(), labels=by_label.keys(), loc='lower left')

    ax.set_title(f"Cluster {frame_idx} and its Itemsets \n Pre 30 Years (1965-1994)")

# -----------------------
# 建立動畫
# -----------------------
fig = plt.figure(figsize=(12, 8))
ax = plt.axes(projection=ccrs.Robinson())

ani = animation.FuncAnimation(fig, update, frames=len(clusters), repeat=True)

# 存成 GIF（需要 Pillow）
ani.save("D:\\es\\visualization\\clusters_animation_post30y.gif", writer="pillow", fps=1)

# 如果要 MP4（需要 ffmpeg）
# ani.save("clusters_animation.mp4", writer="ffmpeg", fps=1)

plt.show()