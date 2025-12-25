import chromadb
from openai import OpenAI
import json
import pandas as pd

API_KEY = ""

client_llm = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=API_KEY,
)

class FPGrowthAlignmentAgent:
    def __init__(self, db_path="./geo_rag_db"):
        self.chroma_client = chromadb.PersistentClient(path=db_path)
        self.collection = self.chroma_client.get_collection(name="geo_rules_cl11")

    def retrieve_historical_rules(self, reasoning_result):
            """
            修正版：解析初步規則，利用特徵相似性進行『偽共現』檢索。
            """
            try:
                # 1. 解析初步推理出的 JSON
                data = json.loads(reasoning_result)
                # 假設 JSON 中包含 rules 列表，取出第一個規則的座標作為關鍵詞
                rules = data.get('rules', [])
                if rules:
                    target_ante = rules[0].get('ante_key', "")
                    target_conse = rules[0].get('conse_key', "")
                    query_text = f"位置 {target_ante} 與 {target_conse} 的地理關聯規律"
                else:
                    query_text = str(reasoning_result)
            except:
                query_text = str(reasoning_result)

            # 2. 執行檢索，同時要求回傳 metadatas (包含我們做好的 s_norm, l_norm 等)
            results = self.collection.query(
                query_texts=[query_text], 
                n_results=5,
                include=['documents', 'metadatas']
            )

            context = ""
            if results['documents'] and len(results['documents'][0]) > 0:
                for i, (doc, meta) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
                    # 注入特徵工程後的數值，讓 Stage 2 的專家 Agent 能判斷這筆歷史資料的強度
                    context += f"【歷史真實規則 {i+1}】：{doc} (特徵強度: {meta})\n"
            else:
                context = "⚠️ 未找到精確座標匹配。請基於地理特徵相似性進行『偽共現』推理，判斷初步計算是否符合該區域的物理邏輯。\n"
                
            return context

    def run_stage_1_reasoning(self, raw_data_str):
        system_prompt = """
        你是一位地理數據科學家。請掃描原始交易資料並執行 FP-Growth 推理。
        1. 計算規則：Support=共現/總數, Confidence=共現/前項數, Lift=Conf/後項支持度。
        2. 物理門檻：僅保留 Lift > 1.2 的規則。
        3. 空間分類：
        - 距離 > 2500km -> Teleconnection
        - 距離 < 2500km -> Regional Diffusion
        請以 JSON 格式回傳，確保數據純淨，不含解釋文字。
        """
        response = client_llm.chat.completions.create(
            model="openai/gpt-4o",
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": raw_data_str}],
            response_format={ "type": "json_object" }
        )
        return response.choices[0].message.content

    def run_stage_2_reflecting(self, raw_data_str, reasoning_result):
        historical_evidence = self.retrieve_historical_rules(reasoning_result)
        
        system_prompt = """
        你是一位精準的數據稽核專家。請比對『初步計算』與『真實規則庫』並產出最終修正結果。

        1. 強制對齊邏輯：
           - 若初步計算的座標與真實規則庫中的座標「高度相似」或「一致」，必須優先採用『真實規則庫』的數值（Support, Confidence, Lift）。
           - 若初步計算的規則在真實庫中完全找不到，且 Lift 過低 (< 1.5)，請將其剔除。
        2. 偽共現推理：
           - 若座標不完全一致但屬於鄰近區域，請參考真實規則的數值水準調優初步結果。
        3. 輸出規範：
           - 必須回傳 JSON 對象，包含：
             - "status": "fixed"
             - "final_rules": [修正後的規則列表，格式同 Stage 1]
             - "reflection_note": [解釋修正了哪些地方]
        """

        user_prompt = f"""
        【初步計算結果】：{reasoning_result}
        【歷史真實參考】：{historical_evidence}
        【原始交易資料參考】：{raw_data_str}
        
        請根據歷史證據，對初步結果進行修正並產出最終 rules。
        """

        response = client_llm.chat.completions.create(
            model="anthropic/claude-3.5-sonnet", # 推薦用邏輯更強的模型進行修正
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            response_format={ "type": "json_object" }
        )
        return response.choices[0].message.content

# --- 執行流程 ---
df = pd.read_csv('data/Hierarchical clustering/分群篩選後資料/pre/transactions_cluster_95threshold_pre30y_cluster11.csv')
raw_data_str = df.to_string()

agent = FPGrowthAlignmentAgent()

print("--- 步驟 1: 原始數據推理 ---")
initial_reasoning = agent.run_stage_1_reasoning(raw_data_str)
print(initial_reasoning)

print("--- 步驟 2 & 3: 檢索、比對、反思修正 ---")
final_report = agent.run_stage_2_reflecting(raw_data_str, initial_reasoning)

print(final_report)