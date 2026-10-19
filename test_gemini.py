from google import genai
from dotenv import load_dotenv
import os
import time

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

start = time.time()

response = client.models.generate_content(
    model="gemini-3.8-flash",
    contents="Say hello."
)

print(response.text)
print(f"Took {time.time()-start:.2f} seconds")