import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# -----------------------
# 讀取資料
# -----------------------
frequent_itemsets_df = pd.read_csv('D:\\es\\frequent_itemset\\frequent_itemsets_95threshold_pre30y.csv')
# frequent_itemsets_df = frequent_itemsets_df.sample(300, random_state=42)
clusters_df = pd.read_csv('D:\\es\\consensus_clustering\\community_assignment_95threshold_pre30y.csv')

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

# 解析節點 & centroid
frequent_itemsets_df['nodes_parsed'] = frequent_itemsets_df['itemset'].apply(parse_nodes)
frequent_itemsets_df['centroid'] = frequent_itemsets_df['nodes_parsed'].apply(compute_centroid)

clusters['nodes_parsed'] = clusters['nodes'].apply(lambda x: [n.strip() for n in x])
clusters['centroid'] = clusters['nodes_parsed'].apply(compute_centroid)

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

best_matches = []
for idx, row in frequent_itemsets_df.iterrows():
    best_idx, best_ratio = find_best_overlap(row['nodes_parsed'], clusters)
    best_matches.append((best_idx, best_ratio))
frequent_itemsets_df['best_cluster_idx'] = [x[0] for x in best_matches]
frequent_itemsets_df['best_overlap'] = [x[1] for x in best_matches]

# -----------------------
# 畫地圖
# -----------------------
fig = plt.figure(figsize=(15,10))
ax = plt.axes(projection=ccrs.Robinson())
ax.add_feature(cfeature.LAND, facecolor='lightgray')
ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS, linestyle=':')
ax.set_extent([-180, 180, 0, 90], crs=ccrs.PlateCarree())

# 畫 cluster 節點 & centroid
for idx, row in clusters.iterrows():
    lats, lons = latlon_from_nodes(row['nodes_parsed'])
    # ax.scatter(lons, lats, color='blue', alpha=0.5, s=20, transform=ccrs.PlateCarree())
    if row['centroid'] is not None:
        ax.scatter(row['centroid'][1], row['centroid'][0], color='darkblue', s=40, transform=ccrs.PlateCarree(), label='Cluster centroid' if idx == 0 else "")

# 畫 itemset 節點 & centroid
itemset_label_added = False
for _, row in frequent_itemsets_df.iterrows():
    if row['centroid'] is not None:
        ax.scatter(
            row['centroid'][1], row['centroid'][0],
            color='darkred', s=40, transform=ccrs.PlateCarree(),
            label='Itemset centroid' if not itemset_label_added else ""
        )
        itemset_label_added = True

# 畫線連結 itemset 與對應 cluster
for idx, row in frequent_itemsets_df.iterrows():
    best_idx = row['best_cluster_idx']
    if best_idx is not None:
        c_itemset = row['centroid']
        c_cluster = clusters.loc[best_idx, 'centroid']
        if c_itemset is not None and c_cluster is not None:
            ax.plot([c_itemset[1], c_cluster[1]], [c_itemset[0], c_cluster[0]],
                    color='gray', alpha=0.4, transform=ccrs.PlateCarree())

ax.legend()
plt.title('Itemsets vs Clusters with Centroid Correspondence \n Pre 30 Years (1965-1994)')
plt.show()
