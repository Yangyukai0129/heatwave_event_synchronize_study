import chromadb
from openai import OpenAI
import json
import pandas as pd

# --- 1. OpenRouter 配置 ---
# 在 OpenRouter 官網申請 API Key
OPENROUTER_API_KEY = ""

client_llm = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

class FPGrowthReflectingAgent:
    def __init__(self, db_path="./fp_growth_db"):
        self.chroma_client = chromadb.PersistentClient(path=db_path)
        self.collection = self.chroma_client.get_collection(name="coordinate_rules_cluster11")

    def retrieve_historical_rules(self, transactions, top_k=5):
        query_text = " ".join(transactions)
        results = self.collection.query(
            query_texts=[query_text],
            n_results=top_k
        )
        context = ""
        if results['documents']:
            for i, doc in enumerate(results['documents'][0]):
                context += f"歷史規則 {i+1}: {doc}\n"
        return context

    def run_analysis(self, user_transactions):
        historical_evidence = self.retrieve_historical_rules(user_transactions)

        # 增加 Reflecting (反思) 邏輯的 System Prompt
        system_prompt = """
        你是一位具備高度自我批判能力的地理數據科學家。
        請遵循以下「思考-檢索-反思-修正」流程：

        1. 【初步推理】：根據輸入 Transaction 計算潛在的 ante->conse 關聯。
        2. 【檢索比對】：對照提供的「歷史規則庫」。
        3. 【自我反思 (Reflection)】：
           - 檢查：我算出的 Lift 與歷史數據是否有顯著差異？
           - 質疑：若附近沒有同步事件，是否因為目前數據量不足以支撐該規則？
           - 判斷：歷史規則與當前數據哪個更具備共識代表性？
        4. 【最終輸出】：給出經過反思後的精確 JSON 報告。
        """

        user_prompt = f"""
        【目前交易資料】：{user_transactions}
        
        【歷史規則庫檢索結果】：
        {historical_evidence}
        
        請分析並回傳 JSON 格式結果，欄位包含：
        ante, conse, support, confidence, lift, status, reflection_note(你的反思紀錄)
        """

        # 呼叫 OpenRouter 上的模型 (例如 claude-3-sonnet 或 gpt-4o)
        response = client_llm.chat.completions.create(
            extra_headers={
                "HTTP-Referer": "http://localhost:3000", # 選填，OpenRouter 統計用
                "X-Title": "FP-Growth RAG Agent",        # 選填
            },
            model="openai/gpt-4o", # 也可以換成 "anthropic/claude-3-sonnet"
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={ "type": "json_object" }
        )

        return response.choices[0].message.content

# --- 執行 ---
current_transactions = pd.read_csv('data/Hierarchical clustering/分群篩選後資料/pre/transactions_cluster_95threshold_pre30y_cluster11.csv')
agent = FPGrowthReflectingAgent()
result = agent.run_analysis(current_transactions)
print(result)
