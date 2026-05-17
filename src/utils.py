import json, re

def clean_text(text: str):
    text_cleaned = re.sub(r'\s+', ' ', text).strip()
    return text_cleaned


def loadData(filePath: str, default_type=list):
    try:
        with open(filePath, 'r', encoding='utf-8') as f:
            content = json.load(f)
            
            if (isinstance(content, dict)):
                return content.get("data", [])
            else:
                return content
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Erro ao carregar dados de {filePath}: {e}")
        return default_type()




