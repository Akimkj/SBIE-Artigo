import os
from src.evaluators.modernBert_eva import modernBertEvaluation
from src.charts import run_charts
from src.utils import loadData
from src.services.openai_service import callApiOpenai
from src.services.deepseek_service import callApiDeepseek
from src.services.claude_service import callApiClaude
from src.services.gemini_service import callApiGemini
from src.count_tokens import count_tokens

# Caminhos para depuração do BertScore
PATH_GEMINIBERT_DEBUG = os.path.join("results", "csv", "debug", "avBert_gemini_debug.csv")
PATH_CLAUDEBERT_DEBUG = os.path.join("results", "csv", "debug", "avBert_claude_debug.csv")
PATH_DEEPSEEKBERT_DEBUG = os.path.join("results", "csv", "debug", "avBert_deepseek_debug.csv")
PATH_OPENAIBERT_DEBUG = os.path.join("results", "csv", "debug", "avBert_openai_debug.csv")
PATH_GEMINIBERT_LIMITED = os.path.join("results", "csv", "limited", "avBert_gemini_limited.csv")
PATH_CLAUDEBERT_LIMITED = os.path.join("results", "csv", "limited", "avBert_claude_limited.csv")
PATH_DEEPSEEKBERT_LIMITED = os.path.join("results", "csv", "limited", "avBert_deepseek_limited.csv")
PATH_OPENAIBERT_LIMITED = os.path.join("results", "csv", "limited", "avBert_openai_limited.csv")
PATH_STATS_CSV = os.path.join("results", "stats", "bert_statistics.csv")
# ── Caminhos dos datasets ──
PATH_GOLDENSET = os.path.join("data", "raw", "stackoverflow_dataset.json")
PATH_GEMINISET  = os.path.join("data", "processed", "base", "gemini_dataset.json")
PATH_CLAUDESET  = os.path.join("data", "processed", "base", "claude_dataset.json")
PATH_DEEPSEEKSET  = os.path.join("data", "processed", "base", "deepseek_dataset.json")
PATH_OPENAISET  = os.path.join("data", "processed", "base", "openai_dataset.json")
PATH_GEMINISET_LIMITED = os.path.join("data", "processed", "limited", "gemini_dataset_limited.json")
PATH_CLAUDESET_LIMITED = os.path.join("data", "processed", "limited", "claude_dataset_limited.json")
PATH_DEEPSEEKSET_LIMITED = os.path.join("data", "processed", "limited", "deepseek_dataset_limited.json")
PATH_OPENAISET_LIMITED =  os.path.join("data", "processed", "limited", "openai_dataset_limited.json")

'''modernBertEvaluation(loadData(PATH_GEMINISET_LIMITED), loadData(PATH_GOLDENSET), PATH_GEMINIBERT_LIMITED)
modernBertEvaluation(loadData(PATH_CLAUDESET_LIMITED), loadData(PATH_GOLDENSET), PATH_CLAUDEBERT_LIMITED)
modernBertEvaluation(loadData(PATH_OPENAISET_LIMITED), loadData(PATH_GOLDENSET), PATH_OPENAIBERT_LIMITED)
modernBertEvaluation(loadData(PATH_DEEPSEEKSET_LIMITED), loadData(PATH_GOLDENSET), PATH_DEEPSEEKBERT_LIMITED)'''


PATH_DIR_ORIGIN = os.path.join("results", "csv", "base")
run_charts(PATH_STATS_CSV, PATH_DIR_ORIGIN)


'''PATH_DIR_LIMITED = os.path.join("results", "csv", "limited")
run_charts(PATH_STATS_CSV, PATH_DIR_LIMITED)'''

'''PATH_DIR_DEBUG = os.path.join("results", "csv")
run_charts(PATH_STATS_CSV, PATH_DIR_DEBUG)'''

'''with open(PATH_GOLDENSET, 'r', encoding='utf-8') as file:
    rawData = json.load(file)

rawData['data'].sort(key=lambda x: len(x["answer"]))

with open(PATH_GOLDENSET, 'w', encoding='utf-8') as file:
    json.dump(rawData,file, indent=4, ensure_ascii=False)'''



