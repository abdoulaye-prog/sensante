# notebooks/test_groq.py
# Test de l'API Groq avec differentes temperatures

import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    print("ERREUR : GROQ_API_KEY non trouvee dans .env")
    exit()

client = Groq(api_key=api_key)

# Test avec temperature = 0.0
print("\n" + "="*50)
print("TEMPERATURE = 0.0 (tres deterministe)")
print("="*50)
response = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[
        {"role": "system", "content": "Tu es un assistant medical. Reponds en francais."},
        {"role": "user", "content": "Explique le paludisme en 2 phrases."}
    ],
    max_tokens=100,
    temperature=0.0
)
print(f"Reponse : {response.choices[0].message.content}")

# Test avec temperature = 0.5
print("\n" + "="*50)
print("TEMPERATURE = 0.5 (equilibre)")
print("="*50)
response = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[
        {"role": "system", "content": "Tu es un assistant medical. Reponds en francais."},
        {"role": "user", "content": "Explique le paludisme en 2 phrases."}
    ],
    max_tokens=100,
    temperature=0.5
)
print(f"Reponse : {response.choices[0].message.content}")

# Test avec temperature = 1.0
print("\n" + "="*50)
print("TEMPERATURE = 1.0 (plus creatif)")
print("="*50)
response = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[
        {"role": "system", "content": "Tu es un assistant medical. Reponds en francais."},
        {"role": "user", "content": "Explique le paludisme en 2 phrases."}
    ],
    max_tokens=100,
    temperature=1.0
)
print(f"Reponse : {response.choices[0].message.content}")