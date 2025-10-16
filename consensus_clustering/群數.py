import pandas as pd
import networkit as nk
from tqdm import tqdm

# ------------------------------
# 1️⃣ 讀取共識矩陣
# ------------------------------
df = pd.read_csv("consensus_clustering/consensus_matrix_from_events_95threshold_detrend_post30y.csv", index_col=0)

# ------------------------------
# 2️⃣ 建圖（只保留共識值 > 0.5 的邊）
# ------------------------------
nodes = list(df.index)
node_to_idx = {node: i for i, node in enumerate(nodes)}
n_nodes = len(nodes)

G = nk.Graph(n_nodes, weighted=True, directed=False)

print("建立圖邊...")
for i in tqdm(range(n_nodes), desc="遍歷節點 (建圖)"):
    for j in range(i + 1, n_nodes):
        w = df.iloc[i, j]
        if w >= 0.5:  # 閾值可調
            G.addEdge(node_to_idx[nodes[i]], node_to_idx[nodes[j]], w)

print(f"圖的節點數: {G.numberOfNodes()}, 邊數: {G.numberOfEdges()}")

# ------------------------------
# 3️⃣ PLM 社群偵測 (一次性檢查)
# ------------------------------
plmCommunities = nk.community.detectCommunities(G, algo=nk.community.PLM(G, refine=True, gamma=1.5))

print("社群數量:", plmCommunities.numberOfSubsets())
print("節點數量:", plmCommunities.numberOfElements())
print("最大社群大小:", max(plmCommunities.subsetSizes()))
print("最小社群大小:", min(plmCommunities.subsetSizes()))
print("平均社群大小:", sum(plmCommunities.subsetSizes()) / plmCommunities.numberOfSubsets())

# 如果想看模組度
mod = nk.community.Modularity().getQuality(plmCommunities, G)
print("Modularity:", mod)

# ------------------------------
# 4️⃣ 輸出每群資料
# ------------------------------

# 方式 1: 每個節點對應群號
node_to_comm = {nodes[i]: plmCommunities.subsetOf(i) for i in range(n_nodes)}
df_node_to_comm = pd.DataFrame(list(node_to_comm.items()), columns=["latlon", "community"])
df_node_to_comm.to_csv("consensus_clustering/community_assignment_95threshold_detrend_post30y.csv", index=False)
print("✅ 已輸出每個節點的社群分配 community_assignment.csv")