import pandas as pd
from collections import Counter
import ast
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.family'] = ['Microsoft JhengHei']  # 或 'SimHei'、'Noto Sans CJK TC'
plt.rcParams['axes.unicode_minus'] = False  # 負號正常顯示

# -----------------------------
# 1. 讀取事件資料
# -----------------------------
df_events = pd.read_csv('events_transactions_95threshold_post30y.csv')
df_events['nodes'] = df_events['nodes'].apply(ast.literal_eval)

# -----------------------------
# 2. 統計每個格點出現次數
# -----------------------------
all_nodes = [node for nodes in df_events['nodes'] for node in nodes]
node_counts = Counter(all_nodes)

node_freq = pd.DataFrame(node_counts.items(), columns=['node', 'count']).sort_values(by='count', ascending=False)
print("格點出現頻率前10名：")
print(node_freq.head(10))

# -----------------------------
# 3. 繪製格點出現頻率分布
# -----------------------------
plt.figure(figsize=(20,6))
plt.bar(range(len(node_freq)), node_freq['count'], alpha=0.7)
plt.xlabel("格點（依出現頻率排序）")
plt.ylabel("出現次數")
plt.title("所有格點出現次數分布(post30y)")

# 固定 Y 軸範圍（最大值到 160）
plt.ylim(0, 40)
plt.yticks(np.arange(0, 41, 5))  # 每 20 一格可讀性更好

# 累積比例曲線
cum_ratio = np.cumsum(node_freq['count']) / np.sum(node_freq['count'])
plt.twinx()
plt.plot(range(len(node_freq)), cum_ratio, color='red', marker='o', markersize=3)
plt.ylabel("累積比例")
plt.grid(alpha=0.3)
plt.show()

# -----------------------------
# 4. 設定門檻篩選高頻格點
# -----------------------------
# threshold = 5  # 先給一個參考值，可依圖調整
# high_freq_nodes = node_freq[node_freq['count'] >= threshold]['node'].tolist()
# print(f"共有 {len(high_freq_nodes)} 個高頻格點被保留")

# # -----------------------------
# # 5. 過濾事件，只保留高頻格點
# # -----------------------------
# df_filtered = df_events.copy()
# df_filtered['nodes'] = df_filtered['nodes'].apply(lambda lst: [n for n in lst if n in high_freq_nodes])
# df_filtered = df_filtered[df_filtered['nodes'].map(len) > 0]

# df_filtered.to_csv('events_filtered_high_freq_nodes.csv', index=False)
# print("篩選後事件資料完成，準備進行共識矩陣或分群分析")