import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import seaborn as sns

# ======== 可調整參數 ========
csv_path = "data/Hierarchical clustering/階層式分群結果/hierarchical_clustering_result_95threshold_pre30y.csv"   # 你的 CSV 檔案路徑
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

colors = sns.color_palette("husl", len(clusters_to_plot))  # 或 "husl", "Set3", "Paired"

# 5. 畫地圖
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())

ax.add_feature(cfeature.LAND, edgecolor='black', facecolor='lightgray')
ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS, linestyle=':')
ax.set_extent([-180, 180, 0, 90], crs=ccrs.PlateCarree())

gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
gl.xlabels_top = False
gl.ylabels_right = False

# 標籤控制
ante_plotted = False
cons_plotted = False

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