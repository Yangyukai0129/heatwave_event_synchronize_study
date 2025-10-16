import pandas as pd
import itertools
import networkit as nk
import numpy as np
from tqdm import tqdm

# 1️⃣ 讀事件 CSV
df = pd.read_csv('events_transactions_95threshold_detrend_post30y.csv')

# 2️⃣ 生成 edge list（同一事件中所有格點兩兩連線）
edges_dict = {}  # key: (node1, node2), value: weight (出現次數)
for _, row in df.iterrows():
    # 假設 nodes 欄位是字串列表，例如 "[0,1,2]"
    # nodes = eval(row['nodes'])  # 或用 ast.literal_eval 更安全
    # 將每個節點轉成 lat_lon 字串（如果你想用經緯度）
    lats = eval(row['latitudes'])
    lons = eval(row['longitudes'])
    nodes = [f"{lat}_{lon}" for lat, lon in zip(lats, lons)]
    
    for u, v in itertools.combinations(sorted(nodes), 2):
        edges_dict[(u, v)] = edges_dict.get((u, v), 0) + 1

# 3️⃣ 轉成 edge list (u, v, weight)
edges = [(u, v, w) for (u, v), w in edges_dict.items()]

# 4️⃣ 建立節點映射
nodes_map = {}
idx_to_node = {}
idx = 0
for u, v, _ in edges:
    for n in [u, v]:
        if n not in nodes_map:
            nodes_map[n] = idx
            idx_to_node[idx] = n
            idx += 1

# 5️⃣ 建立 Networkit Graph
G = nk.Graph(len(nodes_map), weighted=True, directed=False)
for u, v, w in edges:
    G.addEdge(nodes_map[u], nodes_map[v], w)

print(G.numberOfNodes(), G.numberOfEdges())

# 6️⃣ 多次 PLM
num_runs = 100
results = []
for run in tqdm(range(num_runs)):
    plm = nk.community.PLM(G, refine=True, gamma=1.5)
    plm.run()
    partition = plm.getPartition()
    mod = nk.community.Modularity().getQuality(partition, G)
    results.append((mod, partition))

# 7️⃣ 取前 25 高 modularity 並生成共識矩陣
results_sorted = sorted(results, key=lambda x: x[0], reverse=True)
top_partitions = results_sorted[:25]

n_nodes = G.numberOfNodes()
consensus_matrix = np.zeros((n_nodes, n_nodes), dtype=np.float32)

for mod, part in top_partitions:
    groups = np.array([part[i] for i in range(n_nodes)])
    same_group = groups[:, None] == groups[None, :]
    consensus_matrix += same_group.astype(np.float32)

consensus_matrix /= len(top_partitions)

# 8️⃣ 存檔
consensus_df = pd.DataFrame(consensus_matrix, 
                            index=[idx_to_node[i] for i in range(n_nodes)],
                            columns=[idx_to_node[i] for i in range(n_nodes)])
consensus_df.to_csv("consensus_clustering/consensus_matrix_from_events_95threshold_detrend_post30y.csv")
print("✅ 共識矩陣生成完成")