# notebooks/test_temperature.py
import os
from dotenv import load_dotenv
from groq import Groq

# Charger la clé API
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    print("ERREUR: Clé API non trouvée")
    exit()

# Créer le client
client = Groq(api_key=api_key)

# Le message à envoyer
messages = [
    {"role": "system", "content": "Tu es un assistant medical. Reponds en francais."},
    {"role": "user", "content": "Explique le paludisme en 2 phrases."}
]

# Test 1: Temperature 0.0
print("\n🔵 TEMPERATURE = 0.0 (très précis)")
print("-" * 40)
response = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=messages,
    max_tokens=100,
    temperature=0.0
)
print(response.choices[0].message.content)

# Test 2: Temperature 0.5
print("\n🟢 TEMPERATURE = 0.5 (équilibré)")
print("-" * 40)
response = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=messages,
    max_tokens=100,
    temperature=0.5
)
print(response.choices[0].message.content)

# Test 3: Temperature 1.0
print("\n🟠 TEMPERATURE = 1.0 (créatif)")
print("-" * 40)
response = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=messages,
    max_tokens=100,
    temperature=1.0
)
print(response.choices[0].message.content)