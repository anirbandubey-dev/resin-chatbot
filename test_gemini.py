from google import genai
from config import GEMINI_API_KEY

client = genai.Client(api_key=GEMINI_API_KEY)

for model_name in [
    "models/gemini-2.0-flash",
    "models/gemini-flash-latest",
    "models/gemini-2.5-flash",
    "models/gemini-2.5-flash-lite",
]:
    print(f"\nTesting {model_name}...")
    try:
        response = client.models.generate_content(
            model=model_name,
            contents="Say hello in one sentence."
        )
        print("SUCCESS:", response.text)
        break
    except Exception as e:
        print("FAILED:", e)