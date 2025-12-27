import chromadb
from openai import OpenAI
import json
import pandas as pd
import numpy as np
from collections import Counter
import os
from dotenv import load_dotenv
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

load_dotenv()

# 初始化 LLM Client
client_llm = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    default_headers={
        "HTTP-Referer": "http://localhost:3000", # 必填，可以是隨意網址
        "X-Title": "GeoIntelligentAgent",        # 必填，你的應用名稱
    }
)

class GeoIntelligentAgent:
    def __init__(self, df_final, db_path="./geo_rag_db"):
        """
        df_final: 原始特徵工程後的 DataFrame，用於 PCA 全域繪圖
        """
        self.df_final = df_final
        self.chroma_client = chromadb.PersistentClient(path=db_path)
        self.collection = self.chroma_client.get_collection(name="geo_rules_cl11")
        self.feature_cols = [
            's_norm', 'c_norm', 'l_norm', 
            'ante_lat', 'ante_lon_sin', 'ante_lon_cos',
            'conse_lat', 'conse_lon_sin', 'conse_lon_cos'
        ]

    def analyze_region_traits(self, metadatas):
        """
        計算區域統計特徵，包含：
        1. 平均/最高提升度 (Lift)
        2. Regional Diffusion 與 Teleconnection 的比例
        """
        if not metadatas:
            return "此區域尚無歷史統計特徵。"

        # 1. 提取指標
        lifts = [m.get('l_norm', 0) for m in metadatas]
        
        # 2. 計算分類比例 (基於 metadata 中的 category 欄位)
        # 如果入庫時沒存 category，建議依賴 metadata 中的座標距離計算，或確保 df_final 有此欄位
        categories = [m.get('category', 'Unknown') for m in metadatas]
        cat_counts = Counter(categories)
        total = len(categories)
        
        diffusion_ratio = (cat_counts.get('Regional Diffusion', 0) / total) * 100
        tele_ratio = (cat_counts.get('Teleconnection', 0) / total) * 100
        
        trait_summary = (
            f"【區域統計分析】\n"
            f"- 關聯強度：平均提升度 {np.mean(lifts):.2f}，最強達 {np.max(lifts):.2f}。\n"
            f"- 空間分佈：區域擴散 (Regional Diffusion) 佔 {diffusion_ratio:.1f}%，"
            f"遠距連結 (Teleconnection) 佔 {tele_ratio:.1f}%。\n"
            f"- 穩定度評估：{'規律極其強健' if np.mean(lifts) > 0.6 else '存在中等關聯性'}。"
        )
        return trait_summary

    def plot_geo_pca(self, retrieved_ids):
        """
        PCA 視覺化：標註出全域規則、檢索到的相似規則 (偽共現)
        """
        # 準備 PCA 資料
        X = self.df_final[self.feature_cols].fillna(0)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        pca = PCA(n_components=2)
        components = pca.fit_transform(X_scaled)
        
        plot_df = pd.DataFrame(data=components, columns=['PC1', 'PC2'])
        plot_df['rule_id'] = self.df_final.index.map(lambda x: f"rule_{x}").values
        plot_df['ante_key'] = self.df_final['ante_key'].values
        plot_df['Type'] = 'Background Rules'
        plot_df['Size'] = 1

        # 標註檢索到的規則
        plot_df.loc[plot_df['rule_id'].isin(retrieved_ids), 'Type'] = 'Retrieved (Pseudo-co-occurrence)'
        plot_df.loc[plot_df['rule_id'].isin(retrieved_ids), 'Size'] = 5

        fig = px.scatter(
            plot_df, x='PC1', y='PC2', color='Type', size='Size',
            hover_name='ante_key',
            title="Geographic Feature Space (PCA Visualization)",
            template="plotly_dark",
            color_discrete_map={
                'Background Rules': '#444444',
                'Retrieved (Pseudo-co-occurrence)': '#00FFFF'
            }
        )
        return fig
    
    def plot_geo_map(self, retrieved_ids):
        """
        將檢索到的規則畫在 Cartopy 地圖上。
        """
        # 1. 篩選出被檢索到的規則數據
        retrieved_df = self.df_final[self.df_final.index.map(lambda x: f"rule_{x}").isin(retrieved_ids)]
        
        if retrieved_df.empty:
            print("未找到對應的規則數據。")
            return None

        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())
        ax.add_feature(cfeature.LAND, edgecolor='black', facecolor='lightgray')
        ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS, linestyle=':')
        ax.set_extent([-180, 180, 0, 90], crs=ccrs.PlateCarree())

        # 經緯網格線
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                        linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
        gl.xlabels_top = False
        gl.ylabels_right = False

        for _, row in retrieved_df.iterrows():
            try:
                # --- 關鍵修正：清理字串中的 { } ' 字元 ---
                def clean_key(key_str):
                    return str(key_str).replace("{", "").replace("}", "").replace("'", "").strip()

                ante_clean = clean_key(row['ante_key'])
                conse_clean = clean_key(row['conse_key'])

                # 解析座標
                a_lat, a_lon = map(float, ante_clean.split('_'))
                c_lat, c_lon = map(float, conse_clean.split('_'))
                
                # 繪圖邏輯保持不變
                color = '#ff7f0e' if row['category'] == 'Teleconnection' else '#1f77b4'
                
                # 畫連結線
                plt.plot([a_lon, c_lon], [a_lat, c_lat],
                        color=color, linewidth=2,
                        transform=ccrs.Geodetic(),
                        alpha=0.7)
                
                # 標註點
                plt.plot(a_lon, a_lat, marker='o', color='red', transform=ccrs.Geodetic())
                plt.plot(c_lon, c_lat, marker='x', color='black', transform=ccrs.Geodetic())
                
            except Exception as e:
                print(f"解析座標失敗: {row['ante_key']} -> {e}")
                continue

        plt.title("Spatial Connectivity of Geographic Rules")
        return fig

    def search_and_reason(self, user_query):
        """執行檢索、區域特徵分析與專家推理"""
        
        results = self.collection.query(
            query_texts=[user_query],
            n_results=5,
            include=['documents', 'metadatas']
        )
        
        # 1. 計算比例與特徵
        region_traits = self.analyze_region_traits(results['metadatas'][0])
        
        # 2. 準備上下文
        context = ""
        for i, (doc, meta) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
            context += f"【證據 {i+1}】：{doc} (指標: {meta})\n"

        system_prompt = f"""
        你是一位資深地理氣象專家。請參考區域統計背景與具體證據回答問題。
        
        區域背景分析：
        {region_traits}
        
        任務：
        1. 根據 Regional Diffusion 與 Teleconnection 的比例，解釋該區域的地理動力學特性。
        2. 使用『偽共現』邏輯解釋為何檢索到的鄰近規則對此座標有參考意義。
        3. 輸出為 JSON，包含 'expert_analysis' 與 'retrieved_ids'。
        """
        
        response = client_llm.chat.completions.create(
            model="nvidia/nemotron-3-nano-30b-a3b:free",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"問題：{user_query}\n\n檢索證據：\n{context}"}
            ],
            response_format={ "type": "json_object" }
        )
        
        # 回傳包含檢索 ID 以便後續畫圖
        output = json.loads(response.choices[0].message.content)
        output['retrieved_ids'] = results['ids'][0]
        return output

    def audit_and_finalize(self, expert_analysis_dict):
        """第二階段：邏輯稽核"""
        expert_text = expert_analysis_dict.get('expert_analysis', "")
        
        system_prompt = """你是一位數據稽核專家。請審查專家的分析報告，確保其比例分析與物理推論不超過數據強度。"""
        
        response = client_llm.chat.completions.create(
            model="nvidia/nemotron-3-nano-30b-a3b:free",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"待審查報告：{expert_text}"}
            ],
            response_format={ "type": "json_object" }
        )
        return response.choices[0].message.content

# --- 執行範例 ---
# 假設 df_final 已在您的環境中
df_final = pd.read_csv("my_result.csv")
agent = GeoIntelligentAgent(df_final=df_final)

# 1. 專家推理
print("--- 階段 1: 專家檢索與專業解釋 ---")
expert_result = agent.search_and_reason("我想知道 72.625_-102.625 附近的氣象連動規律")
print("專家分析:", expert_result['expert_analysis'])

# 2. 數據稽核
print("\n--- 階段 2: 數據稽核與最終報告 ---")
audit_result = agent.audit_and_finalize(expert_result)
print("稽核報告:", audit_result)

# 3. PCA 視覺化 (在 Jupyter 或 Streamlit 中顯示)
fig = agent.plot_geo_pca(expert_result['retrieved_ids'])
fig.write_html("pca_result.html")
print("圖表已儲存為 pca_result.html，請在資料夾中手動開啟它。")
# fig.show()

# 4. Cartopy 地理視覺化
fig_map = agent.plot_geo_map(expert_result['retrieved_ids'])
if fig_map:
    fig_map.savefig("map_result.png", dpi=300)
    print("地理連結圖已儲存為 map_result.png")