import google.generativeai as genai
import os
from dotenv import load_dotenv
import textwrap
from IPython.display import display
from IPython.display import Markdown

load_dotenv()

GOOGLE_API_KEY=os.getenv("API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

# for m in genai.list_models():
#     if 'generateContent' in m.supported_generation_methods:
#         print(m.name)

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

# response = model.generate_content("boa tarde")
# print(response.text)

chat = model.start_chat(history=[])
prompt = input("Esperando prompt.... ")

while prompt != "fim":
    response = chat.send_message(prompt)
    print("Resposta: ", response.text, "\n")
    prompt = input("Esperando prompt.... ")


# def to_markdown(text):
#   text = text.replace('•', '  *')
#   return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))

# for message in chat.history:
#   display(to_markdown(f'**{message.role}**: {message.parts[0].text}'))
#   print('-------------------------------------------')