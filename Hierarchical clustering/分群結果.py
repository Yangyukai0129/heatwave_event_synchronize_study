import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import squareform
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# -----------------------
# 1. 讀取共識矩陣
# -----------------------
tmp = pd.read_csv(
    'data/consensus_clustering/consensus_matrix_from_events_95threshold_pre30y.csv',
    header=None
)

print(tmp)

# 第一列是欄位名稱（有前導空白或逗號）
columns = tmp.iloc[0, 1:].astype(str).str.strip().tolist()  # 排除第一個空欄，去掉空白
index = tmp.iloc[1:, 0].astype(str).str.strip().tolist()    # 排除第一列的標籤

# 真正的數值矩陣
data = tmp.iloc[1:, 1:].astype(float).values

# 建立 dataframe
consensus_matrix = pd.DataFrame(data, index=index, columns=columns)

# 取前 N 欄，對應 N 個 index
C = consensus_matrix.iloc[:, :len(consensus_matrix)].astype(float).values
print(consensus_matrix.shape)  # 應該是 N x N

# 節點標籤（經緯度字串）
nodes = consensus_matrix.index.tolist()

# ---- 轉距離 ----
D = 1.0 - C
# 若 linkage 需要 condensed distance:
condensed = squareform(D, checks=False)  # 對稱矩陣轉 condensed


# --------------------------
# Hierarchical clustering
# --------------------------
Z = linkage(condensed, method='average')  # 或 'ward' 但 ward 需要原始特徵，這裡用 average 最常見


# --------------------------
# 階層式視覺化
# --------------------------
# import matplotlib.pyplot as plt
# dendrogram(Z)
# plt.show()

# --------------------------
# silhouette分數
# --------------------------
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import fcluster

min_size = 6
best_score, best_k = -1, None

for k in range(2, 200):
    labels = fcluster(Z, k, criterion='maxclust')
    
    # 檢查每個群大小
    cluster_sizes = pd.Series(labels).value_counts()
    if cluster_sizes.min() < min_size:
        continue  # 有群太小就跳過
    
    score = silhouette_score(D, labels, metric='precomputed')
    if score > best_score:
        best_score, best_k = score, k

print(f"最佳群數 k = {best_k}, silhouette = {best_score:.3f}")


# --------------------------
# 根據node把對應cluster存起來
# --------------------------
# k = 14
# labels = fcluster(Z, k, criterion='maxclust')

# cluster_result = pd.DataFrame({
#     'node': nodes,
#     'cluster': labels
# })

# print(cluster_result.head())
# cluster_result.to_csv('data/Hierarchical clustering/階層式分群結果/hierarchical_clustering_result_95threshold_pre30y_min6.csv', index=False)


# --------------------------
# 顯示每個群的平均共識值，並畫出來
# --------------------------
# 計算每個群的平均共識值
# group_means = []
# for g in range(1, k+1):
#     idx = np.where(labels == g)[0]  # 群 g 的事件索引
#     if len(idx) < 2:
#         group_means.append(np.nan)  # 群內只有 1 個事件，平均無意義
#         continue
#     # 群內子矩陣
#     subC = C[np.ix_(idx, idx)]
#     # 取上三角（排除對角線）計算平均
#     triu_idx = np.triu_indices_from(subC, k=1)
#     mean_val = subC[triu_idx].mean()
#     group_means.append(mean_val)

# # 顯示每個群的平均共識值
# for g, val in enumerate(group_means, 1):
#     print(f"群 {g}: 平均共識值 = {val:.3f}")

# # 畫 bar chart
# plt.figure(figsize=(6,4))
# plt.bar(range(1, k+1), group_means, color='skyblue')
# plt.xlabel('Cluster')
# plt.ylabel('Mean Consensus Value')
# plt.title('Mean Consensus Value within Cluster')
# # 在 y=0.7 畫虛線
# plt.axhline(y=0.7, color='red', linestyle='--', linewidth=1.5)
# plt.show()