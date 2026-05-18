# api/main.py
from fastapi import FastAPI
from pydantic import BaseModel, Field
import os
from dotenv import load_dotenv
from groq import Groq

# Charger les variables d'environnement
load_dotenv()

# Client Groq (charge au demarrage)
groq_client = None
groq_api_key = os.getenv("GROQ_API_KEY")

if groq_api_key:
    groq_client = Groq(api_key=groq_api_key)
    print("✅ Client Groq initialise.")
else:
    print("⚠️ ATTENTION : GROQ_API_KEY non trouvee. /explain sera desactive.")
# --- Schemas Pydantic ---
class PatientInput(BaseModel):
    """Donnees d'entree : les symptomes d'un patient."""
    age: int = Field(..., ge=0, le=120, description="Age en annees")
    sexe: str = Field(..., description="Sexe : M ou F")
    temperature: float = Field(..., ge=35.0, le=42.0, description="Temperature en Celsius")
    tension_sys: int = Field(..., ge=60, le=250, description="Tension systolique")
    toux: bool = Field(..., description="Presence de toux")
    fatigue: bool = Field(..., description="Presence de fatigue")
    maux_tete: bool = Field(..., description="Presence de maux de tete")
    region: str = Field(..., description="Region du Senegal")

class DiagnosticOutput(BaseModel):
    """Donnees de sortie : le resultat du diagnostic."""
    diagnostic: str = Field(..., description="Diagnostic predit")
    probabilite: float = Field(..., description="Probabilite du diagnostic")
    confiance: str = Field(..., description="Niveau de confiance")
    message: str = Field(..., description="Recommandation")

class ExplainInput(BaseModel):
    diagnostic: str
    probabilite: float
    age: int
    sexe: str
    temperature: float
    region: str

class ExplainOutput(BaseModel):
    explication: str
    modele_llm: str = "llama-3.1-8b-instant"
# --- Application FastAPI ---
app = FastAPI(
    title="SenSante API",
    description="Assistant pre-diagnostic medical pour le Senegal",
    version="0.2.0"
)
import joblib
import numpy as np
from fastapi.middleware.cors import CORSMiddleware

# Autoriser les requetes depuis le frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# --- Chargement du modele (une seule fois) ---
print("Chargement du modele...")
model = joblib.load("models/model.pkl")
le_sexe = joblib.load("models/encoder_sexe.pkl")
le_region = joblib.load("models/encoder_region.pkl")
feature_cols = joblib.load("models/feature_cols.pkl")
print(f"Modele charge : {type(model).__name__}")
print(f"Classes : {list(model.classes_)}")
@app.get("/health")
@app.get("/model-info")
def model_info():
    """Informations sur le modele charge."""
    return {
        "type": type(model).__name__,
        "nombre_arbres": model.n_estimators,
        "classes": list(model.classes_),
        "nombre_features": model.n_features_in_
    }
@app.post("/predict", response_model=DiagnosticOutput)
def predict(patient: PatientInput):
    """Predire un diagnostic a partir des symptomes d'un patient."""
    
    # 1. Encoder les variables categoriques
    try:
        sexe_enc = le_sexe.transform([patient.sexe])[0]
    except ValueError:
        return DiagnosticOutput(
            diagnostic="erreur", probabilite=0.0,
            confiance="aucune",
            message=f"Sexe invalide : {patient.sexe}. Utiliser M ou F."
        )

    try:
        region_enc = le_region.transform([patient.region])[0]
    except ValueError:
        return DiagnosticOutput(
            diagnostic="erreur", probabilite=0.0,
            confiance="aucune",
            message=f"Region inconnue : {patient.region}"
        )

    # 2. Construire le vecteur de features
    features = np.array([[
        patient.age, sexe_enc, patient.temperature,
        patient.tension_sys, int(patient.toux),
        int(patient.fatigue), int(patient.maux_tete),
        region_enc
    ]])

    # 3. Predire
    diagnostic = model.predict(features)[0]
    proba_max = float(model.predict_proba(features)[0].max())

    # 4. Determiner le niveau de confiance
    confiance = ("haute" if proba_max >= 0.7
                 else "moyenne" if proba_max >= 0.4
                 else "faible")

    # 5. Generer la recommandation
    messages = {
        "palu": "Suspicion de paludisme. Consultez un medecin rapidement.",
        "grippe": "Suspicion de grippe. Repos et hydratation recommandes.",
        "typh": "Suspicion de typhoide. Consultation medicale necessaire.",
        "sain": "Pas de pathologie detectee. Continuez a surveiller."
    }

    # 6. Renvoyer le resultat
    return DiagnosticOutput(
        diagnostic=diagnostic,
        probabilite=round(proba_max, 2),
        confiance=confiance,
        message=messages.get(diagnostic, "Consultez un medecin.")
    )
SYSTEM_PROMPT = """Tu es un assistant medical senegalais.
Tu parles en wolof simple, melange avec du francais si necessaire.
Voici quelques mots wolof :
- Comment ça va ? = Na nga def ?
- Fievre = feebar
- Consulter un medecin = demndeetu doktoor
- Paludisme = feebar bii

Explique le resultat en wolof simple.
Sois rassurant mais recommande toujours une consultation.
Maximum 3 phrases.
Ne fais JAMAIS de diagnostic toi-meme.
Exemple : "Na nga def. Feebar bii la (paludisme) ci 72%. Demndeetu doktoor pour test TDR."""

@app.post("/explain", response_model=ExplainOutput)
def explain(data: ExplainInput):

    """Expliquer un diagnostic en francais avec un LLM."""
    
    if not groq_client:
        return ExplainOutput(
            explication="Service d'explication indisponible. Cle API non configuree.",
            modele_llm="aucun"
        )
    
    # Construire le user prompt
    user_prompt = (
        f"Patient : {data.sexe}, {data.age} ans, region {data.region}\n"
        f"Temperature : {data.temperature} C\n"
        f"Diagnostic du modele : {data.diagnostic} (probabilite {data.probabilite:.0%})\n"
        f"Explique ce resultat au patient."
    )
    
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=200,
            temperature=0.3
        )
        explication = response.choices[0].message.content
    except Exception as e:
        explication = f"Erreur lors de l'appel au LLM : {str(e)}"
    
    return ExplainOutput(explication=explication)
def health_check():
    return {"status": "ok", "message": "SenSante API is running"}