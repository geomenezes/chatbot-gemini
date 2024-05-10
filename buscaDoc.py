import google.generativeai as genai
import os
from dotenv import load_dotenv
import numpy as np
import pandas as pd

load_dotenv()

GOOGLE_API_KEY=os.getenv("API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

model = "models/embedding-001"

DOCUMENT1 = {
    "title": "Operating the Climate Control System",
    "content": "Your Googlecar has a climate control system that allows you to adjust the temperature and airflow in the car. To operate the climate control system, use the buttons and knobs located on the center console.  Temperature: The temperature knob controls the temperature inside the car. Turn the knob clockwise to increase the temperature or counterclockwise to decrease the temperature. Airflow: The airflow knob controls the amount of airflow inside the car. Turn the knob clockwise to increase the airflow or counterclockwise to decrease the airflow. Fan speed: The fan speed knob controls the speed of the fan. Turn the knob clockwise to increase the fan speed or counterclockwise to decrease the fan speed. Mode: The mode button allows you to select the desired mode. The available modes are: Auto: The car will automatically adjust the temperature and airflow to maintain a comfortable level. Cool: The car will blow cool air into the car. Heat: The car will blow warm air into the car. Defrost: The car will blow warm air onto the windshield to defrost it."}
DOCUMENT2 = {
    "title": "Touchscreen",
    "content": "Your Googlecar has a large touchscreen display that provides access to a variety of features, including navigation, entertainment, and climate control. To use the touchscreen display, simply touch the desired icon.  For example, you can touch the \"Navigation\" icon to get directions to your destination or touch the \"Music\" icon to play your favorite songs."}
DOCUMENT3 = {
    "title": "Shifting Gears",
    "content": "Your Googlecar has an automatic transmission. To shift gears, simply move the shift lever to the desired position.  Park: This position is used when you are parked. The wheels are locked and the car cannot move. Reverse: This position is used to back up. Neutral: This position is used when you are stopped at a light or in traffic. The car is not in gear and will not move unless you press the gas pedal. Drive: This position is used to drive forward. Low: This position is used for driving in snow or other slippery conditions."}

documents = [DOCUMENT1, DOCUMENT2, DOCUMENT3]

df = pd.DataFrame(documents)
df.columns = ["Titulo", "Conteudo"]

def embed_fn(title, text):
    return genai.embed_content(model=model, 
                                 content=text, 
                                 title=title, 
                                 task_type="retrieval_document")["embedding"]

# lambda - aplica para cada linha a função
df["Embeddings"] = df.apply(lambda row: embed_fn(row["Titulo"], row["Conteudo"]), axis=1)

def gerar_e_buscar_consulta(consulta, model):
    # pegar consulta
    embedding_da_consulta = genai.embed_content(model=model, 
                                 content=consulta,
                                 task_type="retrieval_query")["embedding"]
    
    # embedding - calculo de distancia da consulta em relação aos docs
    produtos_escalares = np.dot(np.stack(df["Embeddings"]), embedding_da_consulta)
    
    # buscar argumento com maior similaridade e gerar indice do token
    indice = np.argmax(produtos_escalares)

    # localizar o indice
    return df.iloc[indice]["Conteudo"]


consulta = "How do I shift gears in a google car?"

trecho = gerar_e_buscar_consulta(consulta, model)

generation_config = {
  "temperature": 0,
  "candidate_count": 1
}

prompt = f"Reescreva esse texto de uma forma mais descontraída, sem adicionar informações que não façam parte do texto: {trecho}"

model_2 = genai.GenerativeModel("gemini-1.0-pro",
                                generation_config=generation_config)
response = model_2.generate_content(prompt)
print(response.text)
