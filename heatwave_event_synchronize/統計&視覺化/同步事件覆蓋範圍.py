import pandas as pd
import ast
import matplotlib.pyplot as plt

# 讀取前期與後期資料
df_pre = pd.read_csv('events_transactions_95threshold_pre30y.csv')
df_post = pd.read_csv('events_transactions_95threshold_post30y.csv')

# 將 nodes 欄位轉為 list
df_pre['nodes'] = df_pre['nodes'].apply(ast.literal_eval)
df_post['nodes'] = df_post['nodes'].apply(ast.literal_eval)

# 計算每個事件包含的節點數
df_pre['num_nodes'] = df_pre['nodes'].apply(len)
df_post['num_nodes'] = df_post['nodes'].apply(len)

# 新增 period 欄位
df_pre['period'] = 'Early (1965–1994)'
df_post['period'] = 'Late (1995–2024)'

# 合併資料
df_all = pd.concat([df_pre, df_post], ignore_index=True)

# 計算平均覆蓋地點數
mean_cov = df_all.groupby('period')['num_nodes'].mean()
print("平均同步覆蓋地點數：")
print(mean_cov)

# 視覺化比較
plt.figure(figsize=(6,5))
plt.boxplot(
    [df_pre['num_nodes'], df_post['num_nodes']],
    labels=['Early (1965–1994)', 'Late (1995–2024)'],
    patch_artist=True,
    boxprops=dict(facecolor='lightblue'),
    medianprops=dict(color='darkblue')
)
plt.ylabel('Number of synchronized grid points')
plt.title('Spatial Coverage of Synchronization Events')
plt.grid(alpha=0.3)
plt.show()
