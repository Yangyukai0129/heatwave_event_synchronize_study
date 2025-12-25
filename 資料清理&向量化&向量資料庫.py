import ast
import pandas as pd

def normalize_frozenset(x):
    if isinstance(x, str):
        try:
            content = x.replace("frozenset(", "").rstrip(")")
            # 處理單個元素時可能遺失逗號的問題
            if "'" in content and "," not in content:
                items = {content.strip("'")}
            else:
                items = ast.literal_eval(content)
            return sorted(list(items))
        except:
            return [x]
    elif isinstance(x, (frozenset, set)):
        return sorted(list(x))
    return [str(x)]

# 讀取資料
df = pd.read_csv('data/frequent_itemset/pre/association_rules_cluster11.csv')

# 1. 標準化並轉為字串標記
df['ante_list'] = df['antecedents'].apply(normalize_frozenset)
df['conse_list'] = df['consequents'].apply(normalize_frozenset)
df['ante_key'] = df['ante_list'].apply(lambda x: "_".join(x))
df['conse_key'] = df['conse_list'].apply(lambda x: "_".join(x))

# 2. 指標篩選 (Lift > 1.5)
filtered_df = df[df['lift'] > 1.5].copy()

# 3. 去重 (De-duplication): 處理 A->B 與 B->A 的對稱問題
# 建立一個唯一的「無向對」標籤，不論方向，只看組合
def make_unique_pair(row):
    pair = sorted([row['ante_key'], row['conse_key']])
    return "|".join(pair)

filtered_df['pair_id'] = filtered_df.apply(make_unique_pair, axis=1)

# 在每一對重複的關係中，只保留 Lift 最高的那一個方向
filtered_df = filtered_df.sort_values(by='lift', ascending=False)
filtered_df = filtered_df.drop_duplicates(subset=['pair_id'], keep='first')

# 4. 容量限制 (Redundancy Control): 限制單個項目的規則數量
# 避免某個熱點座標（如：77.625_97.375）產生數百條規則淹沒向量庫
# 每個 ante_key 只保留 Lift 前 5 高的規則
final_rules = filtered_df.groupby('ante_key').head(5)

print(f"原始規則數: {len(df)}")
print(f"過濾與去重後規則數: {len(final_rules)}")

# 儲存結果備用
# final_rules.to_csv('data/RAG/processed_rules_cluster6.csv', index=False)

import json

def generate_rag_data(df):
    rag_items = []
    
    for _, row in df.iterrows():
        # 格式化座標顯示
        ante_str = ", ".join(row['ante_list'])
        conse_str = ", ".join(row['conse_list'])
        
        # 建立自然語言描述 (用於向量化)
        # 加入 "同步發生"、"關聯" 等關鍵字增加語意權重
        text_content = (
            f"位置 {ante_str} 與位置 {conse_str} 具有同步關聯。 "
            f"當 {ante_str} 出現時，有 {row['confidence']:.2%} 的機率同時觀測到 {conse_str}。 "
            f"其關聯提升度(Lift)為 {row['lift']:.2f}。"
        )
        
        # 建立 Metadata (用於 LLM 驗證與硬性過濾)
        metadata = {
            "ante_key": row['ante_key'],
            "conse_key": row['conse_key'],
            "support": float(row['support']),
            "confidence": float(row['confidence']),
            "lift": float(row['lift']),
            "source": "fp_growth_cluster6"
        }
        
        rag_items.append({
            "text": text_content,
            "metadata": metadata
        })
    
    return rag_items

# 執行轉換
rag_ready_data = generate_rag_data(final_rules)

# 儲存為 JSON 方便後續匯入向量資料庫
# with open('data/RAG/rag_knowledge_base_cluster11.json', 'w', encoding='utf-8') as f:
#     json.dump(rag_ready_data, f, ensure_ascii=False, indent=2)

print(f"成功生成 {len(rag_ready_data)} 條 RAG 知識條目")

# 2. 直接在此處進行 ChromaDB 匯入
import chromadb

# 建立或讀取本地資料庫
client = chromadb.PersistentClient(path="./fp_growth_db")

# 建立 Collection (建議名稱可以跟 cluster 關聯，如 "coordinate_rules_cluster11")
collection = client.get_or_create_collection(name="coordinate_rules_cluster11")

# 3. 批次寫入資料
if rag_ready_data:
    # 準備資料列表
    # 使用 row index 或自定義 UUID 作為 ID
    ids = [f"cluster11_rule_{i}" for i in range(len(rag_ready_data))]
    documents = [item['text'] for item in rag_ready_data]
    metadatas = [item['metadata'] for item in rag_ready_data]

    # 寫入向量資料庫
    # 注意：ChromaDB 有一次寫入的數量限制，如果資料量破萬，建議分批 (batch)
    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas
    )
    
    print(f"✅ 已成功將 {len(rag_ready_data)} 筆規則存入向量資料庫 './fp_growth_db'")
else:
    print("❌ 無有效資料可匯入")

