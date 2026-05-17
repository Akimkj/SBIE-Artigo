from google import genai
from google.genai import types
import os


def callApiGemini(currID: int, question, numTokens) -> str:
    client = genai.Client()
    result = client.models.generate_content(
                model="gemini-2.5-flash",
                config=types.GenerateContentConfig(
                    max_output_tokens=8192,
                    system_instruction=[
                        "You are an expert in Computer Science and Python documentation.", 
                        "Your task is to answer the provided question in a complete, technical, and detailed manner.",
                        "Return ONLY a valid JSON with keys: 'id': an integer representing the identity of the Question-Answer pair; 'question': a string that will be the question provided; 'answer': a string that will be the returned answer.",
                        f"Rules: 1. DO NOT include any text before or after the JSON; 2. The JSON MUST be syntactically valid; 3. The 'id' must be an integer provided in the input. If not provided, use 0; 4. The 'question' must exactly match the given question; 5. In the 'answer' field, provide a clear and complete response to the question, using appropriate technical details where relevant; 6. The response should be as close as possible to the specified number of tokens, staying within a tolerance margin between {int(numTokens * (1 - 0.10))} tokens and {int(numTokens * (1 + 0.10))} tokens. Avoid exceeding this margin unless strictly necessary; 7. Escape all special characters correctly within the JSON strings; 8. DO NOT include Markdown; 9. DO NOT omit any required fields."
                    ]
                ),
                contents=f"ID: {currID}\n Question provided: {question}\n Number of tokens: {numTokens}",
            )
    return result.text or ""