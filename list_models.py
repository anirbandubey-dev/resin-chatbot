from openai import OpenAI
from config import NVIDIA_NIM_API_KEY, NVIDIA_NIM_BASE_URL

client = OpenAI(api_key=NVIDIA_NIM_API_KEY, base_url=NVIDIA_NIM_BASE_URL)

for model in client.models.list().data:
    print(model.id)
