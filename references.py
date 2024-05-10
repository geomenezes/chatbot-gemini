import google.generativeai as genai
import os
from dotenv import load_dotenv
import textwrap
from IPython.display import display
from IPython.display import Markdown

load_dotenv()

GOOGLE_API_KEY=os.getenv("API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

for m in genai.list_models():
    if 'generateContent' in m.supported_generation_methods:
        print(m.name)

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


def to_markdown(text):
  text = text.replace('•', '  *')
  return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))

chat = model.start_chat(history=[])

for message in chat.history:
  display(to_markdown(f'**{message.role}**: {message.parts[0].text}'))
  print('-------------------------------------------')

model = "models/embedding-001"

for m in genai.list_models():
    if 'embedContent' in m.supported_generation_methods:
        print(m.name)

text = "Hello World"
result = genai.embed_content(model=model, content=text)

print(result['embedding'])
print(len(result['embedding']))

title = "The next generation of AI for developers and Google Workspace"
sample_text = ("Title: The next generation of AI for developers and Google Workspace"
    "\n"
    "Full article:\n"
    "\n"
    "Gemini API & Google AI Studio: An approachable way to explore and prototype with generative AI applications")

embeddings = genai.embed_content(model=model, 
                                 content=sample_text, 
                                 title=title, 
                                 task_type="retrieval_document")

print(embeddings)