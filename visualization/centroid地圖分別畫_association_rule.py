import csv
import os
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import ast
import pandas as pd

INPUT_RULES_CSV = "D:\\es\\association_rule\\association_rules_with_centers_post30y.csv"
CLUSTERS_CSV = "D:\\es\\consensus_clustering\\community_assignment_95threshold_post30y.csv"
OUTPUT_DIR = "D:\\es\\visualization\\rules_clustered_filtered"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MIN_ANTE = 1
MIN_CONS = 1
MIN_DISTANCE_KM = 0
MIN_RULES_PER_CLUSTER = 1

# -----------------------
# 輔助函式
# -----------------------
def to_set(item):
    if isinstance(item, str):
        try:
            return set(ast.literal_eval(item))
        except:
            return set()
    return set(item)

def latlon_from_node(n):
    lat, lon = n.split("_")
    return float(lat), float(lon)

def compute_centroid(nodes):
    lats, lons = [], []
    for n in nodes:
        try:
            lat, lon = latlon_from_node(n)
            lats.append(lat)
            lons.append(lon)
        except:
            continue
    if not lats:
        return (np.nan, np.nan)
    return (np.mean(lats), np.mean(lons))

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi, dlambda = np.radians(lat2 - lat1), np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))

def draw_basemap(ax):
    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.set_global()

# -----------------------
# 讀 rules CSV
# -----------------------
rows = []
with open(INPUT_RULES_CSV, newline='', encoding='utf-8') as f:
    reader = csv.reader(f)
    header = next(reader)
    for row in reader:
        rows.append(row)
rules_df = pd.DataFrame(rows, columns=header)

# 讀 clusters
clusters_df = pd.read_csv(CLUSTERS_CSV)

# cluster 節點集合
cluster_points_set = set()
for idx, row in clusters_df.iterrows():
    cluster_points_set.update(to_set(row['latlon']))

# 解析 antecedent / consequent
rules_df['ante_nodes'] = rules_df['antecedents'].apply(to_set)
rules_df['cons_nodes'] = rules_df['consequents'].apply(to_set)
rules_df['ante_size'] = rules_df['ante_nodes'].apply(len)
rules_df['cons_size'] = rules_df['cons_nodes'].apply(len)

# centroid
rules_df['ante_centroid'] = rules_df['ante_nodes'].apply(compute_centroid)
rules_df['cons_centroid'] = rules_df['cons_nodes'].apply(compute_centroid)
rules_df['ante_lat'] = rules_df['ante_centroid'].apply(lambda x: x[0])
rules_df['ante_lon'] = rules_df['ante_centroid'].apply(lambda x: x[1])
rules_df['cons_lat'] = rules_df['cons_centroid'].apply(lambda x: x[0])
rules_df['cons_lon'] = rules_df['cons_centroid'].apply(lambda x: x[1])

# distance
rules_df['distance_km'] = rules_df.apply(
    lambda r: haversine_km(r['ante_lat'], r['ante_lon'], r['cons_lat'], r['cons_lon'])
    if not np.isnan(r['ante_lat']) and not np.isnan(r['cons_lat']) else np.nan, axis=1
)

# 篩選規則
def has_cluster_match(nodes):
    return any(n in cluster_points_set for n in nodes)

mask_valid = (
    (rules_df['ante_size'] >= MIN_ANTE) &
    (rules_df['cons_size'] >= MIN_CONS) &
    (rules_df['distance_km'] >= MIN_DISTANCE_KM) &
    rules_df['ante_nodes'].apply(has_cluster_match) &
    rules_df['cons_nodes'].apply(has_cluster_match)
)
rules_valid = rules_df[mask_valid].copy()
print(f"總規則數: {len(rules_df)}, 符合條件規則數: {len(rules_valid)}")

# -----------------------
# 可視化
# -----------------------
fig = plt.figure(figsize=(14,8))
ax = plt.axes(projection=ccrs.Robinson())
draw_basemap(ax)

# cluster 節點
for idx, row in clusters_df.iterrows():
    nodes = to_set(row['latlon'])
    for n in nodes:
        if any(n in ante or n in cons for ante, cons in zip(rules_valid['ante_nodes'], rules_valid['cons_nodes'])):
            lat, lon = latlon_from_node(n)
            ax.plot(lon, lat, marker='o', markersize=5, color='blue', alpha=0.6, transform=ccrs.PlateCarree())

# antecedent → consequent 線
for idx, row in rules_valid.iterrows():
    a_lat, a_lon = row['ante_lat'], row['ante_lon']
    c_lat, c_lon = row['cons_lat'], row['cons_lon']
    ax.plot([a_lon, c_lon], [a_lat, c_lat], color='gray', alpha=0.4, linewidth=1, transform=ccrs.PlateCarree())

plt.title("Filtered Association Rules with Cluster Match")
plt.show()