from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

def callApiOpenai(id: int, question, numTokens):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {
                "role": "system",
                "content": f"You are a Computer Science and Python documentation expert. Your task is to answer the given question completely, technically, and in detail. Return ONLY a valid JSON with the following keys: 'id': an integer representing the identity of the Question-Answer pair; 'question': a string that will be the given question; 'answer': a string that will be the returned answer. Rules: 1. DO NOT include any text before or after the JSON; 2. The JSON MUST be syntactically valid; 3. The 'id' must be an integer provided in the input. If not provided, use 0; 4. The 'question' must exactly match the given question; 5. In the 'answer' field, provide a clear and complete response to the question, using appropriate technical details where relevant; 6. The response should be as close as possible to the specified number of tokens, staying within a tolerance margin between {int(numTokens * (1 - 0.10))} tokens and {int(numTokens * (1 + 0.10))} tokens. Avoid exceeding this margin unless strictly necessary.; 7. Escape all special characters correctly within the JSON strings; 8. DO NOT include Markdown; 9. DO NOT omit any required fields."
            },
            {
                "role": "user",
                "content": f"ID: {id}\n Question provided: {question}\n Number of tokens: {numTokens}"
            },
        ],
        max_tokens=8192,
    )

    return response.choices[0].message.content or ""