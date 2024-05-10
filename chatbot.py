import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY=os.getenv("API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

generation_config = {
    "candidate_count": 1,
    "temperature": 0.5,
}

safety_settings = {
    "HARASSMENT": "BLOCK_ONLY_HIGH",
    "HATE": "BLOCK_ONLY_HIGH",
    "SEXUAL": "BLOCK_ONLY_HIGH",
    "DANGEROUS": "BLOCK_ONLY_HIGH",
}

model = genai.GenerativeModel(model_name="gemini-1.0-pro",
                              generation_config=generation_config,
                              safety_settings=safety_settings)

chat = model.start_chat(history=[])
prompt = input("Esperando prompt.... ")

while prompt != "fim":
    response = chat.send_message(prompt)
    print("Resposta: ", response.text, "\n")
    prompt = input("Esperando prompt.... ")
