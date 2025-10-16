import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import seaborn as sns

# ======== 可調整參數 ========
csv_path = "Hierarchical clustering/hierarchical_clustering_result_95threshold_pre30y.csv"   # 你的 CSV 檔案路徑
clusters_to_plot = [6,9,11] # 想要畫的 cluster 編號（可自行修改）
# ============================

# 1. 讀取資料
df = pd.read_csv(csv_path)

# 2. 拆分 node 為緯度、經度
df[['lat', 'lon']] = df['node'].str.split('_', expand=True).astype(float)

# 3. 篩選只保留要畫的 cluster
df = df[df['cluster'].isin(clusters_to_plot)]

# 4. 準備顏色
# colors = plt.cm.tab20(np.linspace(0, 1, len(clusters_to_plot)))

# 使用 hsv 色譜產生 30 種顏色
# cmap = plt.cm.get_cmap('hsv', len(clusters_to_plot))
# colors = [cmap(i) for i in range(len(clusters_to_plot))]

colors = sns.color_palette("tab20", len(clusters_to_plot))  # 或 "husl", "Set3", "Paired"

# 5. 畫地圖
plt.figure(figsize=(10, 6))
ax = plt.axes(projection=ccrs.PlateCarree())

# 加底圖
ax.add_feature(cfeature.LAND, facecolor='lightgray')
ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
ax.add_feature(cfeature.BORDERS, linestyle=':')
ax.add_feature(cfeature.COASTLINE)

# 範圍自動根據資料調整
ax.set_extent([-180, 180, 0, 90], crs=ccrs.PlateCarree())

# 6. 逐個 cluster 畫圖
for i, cluster_id in enumerate(clusters_to_plot):
    subset = df[df['cluster'] == cluster_id]
    ax.scatter(subset['lon'], subset['lat'],
               s=20, facecolors='none', edgecolors=colors[i],
               linewidth=1.5, transform=ccrs.PlateCarree(),
              #  label=f'Cluster {cluster_id}'
               )

# plt.legend(loc='lower left', fontsize=9)
plt.title('Selected Clusters on Map(pre30y)', fontsize=14)
plt.show()