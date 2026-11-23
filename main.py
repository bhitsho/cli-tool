from openai import OpenAI
import os
from dotenv import load_dotenv

#Constants
MODEL_NAME = "openai/gpt-oss-20b"
SYSTEM_PROMPT = "You are a helpful assistant."
BASE_URL = "https://api.groq.com/openai/v1"

#loading environment variables
load_dotenv()


client = OpenAI(
    base_url = BASE_URL,
    api_key = os.environ["API_KEY"]
)


response = client.chat.completions.create(
    model=MODEL_NAME,
    messages=[
         {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "Hello"}
    ]
)

print(response)

