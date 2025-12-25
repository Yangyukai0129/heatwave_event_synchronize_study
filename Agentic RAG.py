import chromadb
from openai import OpenAI
import json
import pandas as pd

# --- 1. OpenRouter 配置 ---
# 在 OpenRouter 官網申請 API Key
OPENROUTER_API_KEY = ""

# OpenRouter 配置
client_llm = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

class FPGrowthReflectingAgent:
    def __init__(self, db_path="./fp_growth_db"):
        self.chroma_client = chromadb.PersistentClient(path=db_path)
        self.collection = self.chroma_client.get_collection(name="coordinate_rules_cluster11")

    def retrieve_historical_rules(self, transactions, top_k=5):
        # 這裡建議將 DataFrame 轉為字串進行檢索
        query_text = str(transactions) 
        results = self.collection.query(query_texts=[query_text], n_results=top_k)
        context = ""
        if results['documents']:
            for i, doc in enumerate(results['documents'][0]):
                context += f"歷史規則 {i+1}: {doc}\n"
        return context

    def run_analysis(self, user_transactions):
        historical_evidence = self.retrieve_historical_rules(user_transactions)
        system_prompt = "你是一位地理數據專家。請根據提供數據產出關聯規則 JSON，並執行自我反思。"
        
        user_prompt = f"數據：{user_transactions}\n歷史規則：\n{historical_evidence}\n回傳格式：JSON (ante, conse, support, confidence, lift, status, reflection_note)"

        response = client_llm.chat.completions.create(
            model="openai/gpt-4o",
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            response_format={ "type": "json_object" }
        )
        return response.choices[0].message.content

    def re_fix_analysis(self, user_transactions, wrong_result, critic_feedback):
        """根據稽核員的意見進行修正"""
        historical_evidence = self.retrieve_historical_rules(user_transactions)
        
        system_prompt = "你是一位負責修正數據的專家。請根據稽核員的具體意見，修正之前的錯誤結果。"
        user_prompt = f"""
        【之前的錯誤結果】：{wrong_result}
        【稽核員意見】：{critic_feedback}
        【參考歷史證據】：{historical_evidence}
        
        請重新產出修正後的 JSON 報告。
        """
        
        response = client_llm.chat.completions.create(
            model="openai/gpt-4o",
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            response_format={ "type": "json_object" }
        )
        return response.choices[0].message.content

    def run_analysis_with_loop(self, user_transactions, max_retries=2):
        print("--- 開始第一輪推論 ---")
        current_result = self.run_analysis(user_transactions)
        
        for i in range(max_retries):
            # 2. 稽核員進行審查 (建議用邏輯較強的模型，如 Claude 3.5 Sonnet)
            review_prompt = f"請稽核此結果是否符合歷史規則：{current_result}\n證據：{self.retrieve_historical_rules(user_transactions)}\n有錯請指出，無錯請回傳 OK。"
            
            review_check = client_llm.chat.completions.create(
                model="anthropic/claude-3.5-sonnet",
                messages=[{"role": "user", "content": review_prompt}]
            ).choices[0].message.content

            if "OK" in review_check.upper():
                print(f"--- 稽核通過 (第 {i+1} 輪) ---")
                return current_result
            else:
                print(f"--- 稽核失敗：{review_check} ---")
                current_result = self.re_fix_analysis(user_transactions, current_result, review_check)
        
        return current_result

# --- 執行 ---
df = pd.read_csv('data/Hierarchical clustering/分群篩選後資料/pre/transactions_cluster_95threshold_pre30y_cluster11.csv')
agent = FPGrowthReflectingAgent()
result = agent.run_analysis_with_loop(df)
print(result)
