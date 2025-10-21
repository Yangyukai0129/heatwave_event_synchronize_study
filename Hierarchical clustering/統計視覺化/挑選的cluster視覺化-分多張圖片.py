import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import seaborn as sns

# ======== 可調整參數 ========
csv_path = "data/Hierarchical clustering/階層式分群結果/hierarchical_clustering_result_95threshold_post30y.csv"
clusters_to_plot = [12, 16, 17, 22, 42, 41, 36, 27, 26, 50, 45, 59, 79, 76, 72, 68, 61, 136, 105, 106, 107, 115, 80, 81, 86, 95, 87, 91, 92, 99, 96, 101, 102, 159, 142, 143, 146, 185, 120, 128, 132, 186, 162, 21, 18, 25, 19, 78, 20]
clusters_per_fig = 10   # 每張圖要畫幾個 cluster
# ============================

# 1. 讀取資料
df = pd.read_csv(csv_path)

# 2. 拆分 node 為緯度、經度
df[['lat', 'lon']] = df['node'].str.split('_', expand=True).astype(float)

# 3. 篩選只保留要畫的 cluster
df = df[df['cluster'].isin(clusters_to_plot)]

# 4. 顏色
colors = sns.color_palette("tab10", len(clusters_to_plot))

# 5. 分組繪圖
for start in range(0, len(clusters_to_plot), clusters_per_fig):
    group = clusters_to_plot[start:start + clusters_per_fig]
    group_colors = colors[start:start + clusters_per_fig]
    
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

    # 畫 cluster
    for cluster_id, color in zip(group, group_colors):
        subset = df[df['cluster'] == cluster_id]
        ax.scatter(subset['lon'], subset['lat'],
                   s=20, facecolors='none', edgecolors=color,
                   linewidth=1.5, transform=ccrs.PlateCarree(),
                   label=f'Cluster {cluster_id}')
    
    plt.legend(loc='lower left', fontsize=8, frameon=False)
    plt.title(f'Clusters {group[0]}–{group[-1]} (post30y)', fontsize=14)
    plt.tight_layout()
    plt.show()