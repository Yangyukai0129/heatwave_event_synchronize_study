import chromadb
from openai import OpenAI
import json
import pandas as pd
import numpy as np
from collections import Counter
import os
from dotenv import load_dotenv

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
    def __init__(self, db_path="./geo_rag_db"):
        self.chroma_client = chromadb.PersistentClient(path=db_path)
        self.collection = self.chroma_client.get_collection(name="geo_rules_cl11")

    def analyze_region_traits(self, metadatas):
        """
        分析檢索到的 Metadata，提取該區域的「統計性格」。
        這能幫助 LLM 了解該區是否普遍存在強關聯。
        """
        if not metadatas:
            return "此區域尚無歷史統計特徵。"

        lifts = [m.get('l_norm', 0) for m in metadatas]
        # 假設我們在 metadata 存了原始類別，若無則略過
        avg_lift = np.mean(lifts)
        max_lift = np.max(lifts)
        
        trait_summary = (
            f"區域統計特徵：平均標準化提升度為 {avg_lift:.2f}，"
            f"觀測到最強關聯強度為 {max_lift:.2f}。 "
            f"顯示該區域規律穩定性為 {'高' if avg_lift > 0.5 else '中低'}。"
        )
        return trait_summary

    def search_and_reason(self, user_query):
        """第一階段：檢索 + 區域分析 + 專家解釋"""
        
        # 1. 執行 ChromaDB 檢索
        results = self.collection.query(
            query_texts=[user_query],
            n_results=5,
            include=['documents', 'metadatas']
        )
        
        # 2. 提取區域特徵 (Traits)
        region_traits = self.analyze_region_traits(results['metadatas'][0])
        
        # 3. 格式化檢索內容 (Context)
        context = ""
        for i, (doc, meta) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
            context += f"【證據 {i+1}】：{doc} (數值指標: {meta})\n"

        system_prompt = f"""
        你是一位資深地理氣象專家。請參考區域統計背景與具體規律證據回答問題。
        
        區域背景：{region_traits}
        
        任務：
        1. 解釋地理關聯的物理意義（如遠距連結 Teleconnection）。
        2. 利用『偽共現』邏輯：若精確位置不符，請說明為何這些相似特徵的規則具有參考價值。
        3. 輸出為 JSON 格式，包含 'expert_analysis' 欄位。
        """
        
        response = client_llm.chat.completions.create(
            model="nvidia/nemotron-3-nano-30b-a3b:free",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"使用者問題：{user_query}\n\n檢索證據：\n{context}"}
            ],
            response_format={ "type": "json_object" }
        )
        return response.choices[0].message.content

    def audit_and_finalize(self, expert_analysis):
        """第二階段：數據稽核 (使用邏輯更強的模型如 Claude)"""
        
        system_prompt = """
        你是一位地理數據稽核專家。請審查氣象專家的分析。
        1. 嚴格性：檢查是否有 Lift 或 Confidence 過低卻過度解讀的情況。
        2. 空間邏輯：確認『偽共現』的推論是否符合地理近鄰約束（距離太遠的相似性需謹慎）。
        3. 最終報告：輸出 JSON，包含 'final_decision' (Pass/Fail) 與 'audit_report'。
        """
        
        response = client_llm.chat.completions.create(
            model="nvidia/nemotron-3-nano-30b-a3b:free",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"待審查內容：{expert_analysis}"}
            ],
            response_format={ "type": "json_object" }
        )
        return response.choices[0].message.content

# --- 執行實例 ---
agent = GeoIntelligentAgent()
query = "座標 72.625_-102.625 在極端氣候下的潛在關聯規律為何？"

print("--- 執行：專家推理中 ---")
expert_out = agent.search_and_reason(query)
print(expert_out)

print("\n--- 執行：稽核審查中 ---")
final_report = agent.audit_and_finalize(expert_out)
print(final_report)