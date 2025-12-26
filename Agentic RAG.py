import chromadb
from openai import OpenAI
import json
import pandas as pd

API_KEY = ""

client_llm = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=API_KEY,
)

class GeoKnowledgeAgent:
    def __init__(self, db_path="./geo_rag_db"):
        self.chroma_client = chromadb.PersistentClient(path=db_path)
        # 確保名稱與你入庫時一致
        self.collection = self.chroma_client.get_collection(name="geo_rules_cl11")

    def search_expert_knowledge(self, user_query):
        """根據使用者提問，搜尋地理規律並進行專業解釋"""
        
        # 1. 檢索最相關的地理規則
        results = self.collection.query(
            query_texts=[user_query],
            n_results=3,
            include=['documents', 'metadatas']
        )
        
        # 2. 格式化檢索結果供 LLM 參考
        context = ""
        if results['documents']:
            for i, (doc, meta) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
                context += f"【地理規則實體 {i+1}】：{doc} (特徵指標: {meta})\n"

        # 3. 氣象/地理專家進行解釋
        system_prompt = """
        你是一位資深地理氣象專家。請根據檢索到的地理規則實體，回答使用者的問題。
        你的任務是：
        1. 解釋這些地理關聯背後的物理意義（例如：遠距連結 Teleconnection 或區域擴散）。
        2. 根據提供特徵指標（Lift/Confidence），評估該現象發生的可能性。
        3. 如果找不到完全一致的地點，請利用『偽共現』邏輯，參考最相似的地點特徵給出推論。
        請以專業且易懂的語氣回覆，輸出為 JSON 格式，包含 'expert_analysis'。
        """
        
        response = client_llm.chat.completions.create(
            model="openai/gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"使用者提問：{user_query}\n\n檢索到的背景知識：\n{context}"}
            ],
            response_format={ "type": "json_object" }
        )
        return response.choices[0].message.content

    def audit_and_finalize(self, expert_analysis):
        """第二階段：稽核專家確保結論符合統計強度"""
        
        system_prompt = """
        你是一位數據稽核專家。請審查氣象專家的分析報告。
        1. 嚴格檢查：若 Lift < 2.0，請提醒用戶該關聯性較弱。
        2. 空間邏輯：檢查分析是否符合地理近鄰約束。
        3. 最終校正：確保報告中沒有過度推論。
        輸出 JSON，包含 'final_report' 與 'audit_note'。
        """
        
        response = client_llm.chat.completions.create(
            model="anthropic/claude-3.5-sonnet",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"待審查報告：{expert_analysis}"}
            ],
            response_format={ "type": "json_object" }
        )
        return response.choices[0].message.content

# --- 執行流程 ---
agent = GeoKnowledgeAgent()

# 範例：User 不再給 CSV，而是問一個具體的位置
user_input = "我想知道座標 72.625_-102.625 附近有哪些顯著的氣象關聯規律？"

print("--- 階段 1: 專家檢索與專業解釋 ---")
expert_json = agent.search_expert_knowledge(user_input)
print(expert_json)

print("\n--- 階段 2: 數據稽核與最終報告 ---")
final_json = agent.audit_and_finalize(expert_json)
print(final_json)