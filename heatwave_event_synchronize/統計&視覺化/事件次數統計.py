import pandas as pd
import ast

# 讀取資料
df = pd.read_csv("events_transactions_95threshold_pre30y.csv")

# 將字串轉成 list（如果 nodes 是字串格式）
df["nodes"] = df["nodes"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

# 計算每個事件的節點數
df["num_nodes"] = df["nodes"].apply(len)

print(df[["event_id", "num_nodes"]].head())

# 顯示統計結果
print("最少節點數：", df["num_nodes"].min())
print("最多節點數：", df["num_nodes"].max())
print("平均節點數：", df["num_nodes"].mean())

import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(8, 5))
plt.hist(df["num_nodes"], bins=10, range=(0, 700), edgecolor='black', alpha=0.7)
plt.title("Distribution of Number of Nodes per Event(Pre30y)", fontsize=14)
plt.xlabel("Number of Nodes", fontsize=12)
plt.ylabel("Count of Events", fontsize=12)
plt.grid(alpha=0.3)

# 固定 X 軸範圍與刻度
plt.xlim(0, 700)
plt.xticks(np.linspace(0, 700, 11, dtype=int))

# 固定 Y 軸範圍（最大值到 160）
plt.ylim(0, 180)
plt.yticks(np.arange(0, 181, 20))  # 每 20 一格可讀性更好

plt.show()