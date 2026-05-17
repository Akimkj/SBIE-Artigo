import tiktoken, os
from google import genai
from anthropic import Anthropic
from dotenv import load_dotenv
load_dotenv()

def count_tokens(sentence, model):
    """
    count_tokens: Recebe uma string e conta a quantidade de tokens de acordo com o modelo utilizado.
    """
    if (model == "openai" or model == "deepseek"):
        enconding = tiktoken.get_encoding("cl100k_base")
        tokens = enconding.encode(sentence)
        return len(tokens)
    if (model == "gemini"):
        client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
        res = client.models.count_tokens(
            model="gemini-2.5-flash",
            contents={sentence}
        )
        return res.total_tokens
    if (model == "claude"):
        client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        res = client.messages.count_tokens(
            model="claude-sonnet-4-6",
            messages=[{"role": "user", "content": sentence}]
        )
        return res.input_tokens