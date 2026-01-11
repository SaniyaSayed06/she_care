# ===================== IMPORTS =====================
import os
import joblib
import pandas as pd
import openai

from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import Column, Integer, String, Float, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from passlib.context import CryptContext

# ===================== CONFIG =====================
DATABASE_URL = "sqlite:///./shecare.db"
MODEL_PATH = "pcos_model.pkl"

openai.api_key = os.getenv("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")

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# ===================== DATABASE SETUP =====================
engine = create_engine(
    DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

# ===================== DATABASE MODELS =====================
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    hashed_password = Column(String)

class PatientPrediction(Base):
    __tablename__ = "patients"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String)
    probability = Column(Float)
    risk_label = Column(String)
    advice = Column(String)

Base.metadata.create_all(bind=engine)

# ===================== FASTAPI APP =====================
app = FastAPI(title="SheCare Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===================== LOAD ML MODEL =====================
model = joblib.load(MODEL_PATH)

FEATURES = [
    "Age", "Weight", "Height", "BloodPressure",
    "Insulin", "Glucose", "HairGrowth", "SkinIssues"
]

# ===================== DEPENDENCY =====================
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ===================== PASSWORD UTILS =====================
def hash_password(password: str):
    return pwd_context.hash(password)

def verify_password(password: str, hashed: str):
    return pwd_context.verify(password, hashed)

# ===================== ROOT =====================
@app.get("/")
def root():
    return {"status": "SheCare backend running"}

# ===================== REGISTER =====================
@app.post("/register")
def register(data: dict, db: Session = Depends(get_db)):
    if db.query(User).filter(User.username == data["username"]).first():
        return {"error": "Username already exists"}

    user = User(
        username=data["username"],
        hashed_password=hash_password(data["password"])
    )
    db.add(user)
    db.commit()
    return {"message": "User registered successfully"}

# ===================== LOGIN =====================
@app.post("/login")
def login(data: dict, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == data["username"]).first()

    if not user or not verify_password(data["password"], user.hashed_password):
        return {"error": "Invalid username or password"}

    return {"message": "Login successful"}

# ===================== PREDICT + AI ADVICE =====================
@app.post("/predict_and_advice")
def predict_and_advice(data: dict, db: Session = Depends(get_db)):

    df = pd.DataFrame([data])[FEATURES]

    prob = model.predict_proba(df)[:, 1][0]
    label = "High chances" if prob > 0.5 else "Low chances"

    advice = ""
    if "prompt" in data:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a healthcare assistant for PCOS."},
                {"role": "user", "content": data["prompt"]}
            ],
            temperature=0.7,
            max_tokens=300
        )
        advice = response["choices"][0]["message"]["content"]

    record = PatientPrediction(
        username=data["username"],
        probability=float(prob),
        risk_label=label,
        advice=advice
    )
    db.add(record)
    db.commit()

    return {
        "probability": float(prob),
        "label": label,
        "advice": advice
    }

# ===================== CHATBOT =====================
@app.post("/chatbot")
def chatbot(data: dict):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a friendly PCOS chatbot. Be supportive and simple."},
            {"role": "user", "content": data["message"]}
        ],
        temperature=0.7,
        max_tokens=300
    )
    return {"reply": response["choices"][0]["message"]["content"]}

# ===================== WEEKLY REPORT =====================
@app.post("/weekly_report")
def weekly_report(data: dict):
    prompt = f"""
    Generate a weekly PCOS health report using the following data:
    {data["weekly_data"]}

    Include:
    - Health summary
    - Improvements
    - Concerns
    - Lifestyle tips
    - Encouraging tone
    """

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.6,
        max_tokens=500
    )
    return {"weekly_report": response["choices"][0]["message"]["content"]}

# ===================== EXPLAIN SYMPTOM =====================
@app.post("/explain_symptom")
def explain_symptom(data: dict):
    prompt = f"""
    Explain the PCOS symptom "{data['symptom']}".
    Explanation style: {data['style']}.
    Keep it simple, reassuring, and medically correct.
    """

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
        max_tokens=300
    )
    return {"explanation": response["choices"][0]["message"]["content"]}

# ===================== HISTORY =====================
@app.get("/history/{username}")
def get_history(username: str, db: Session = Depends(get_db)):
    records = db.query(PatientPrediction).filter(
        PatientPrediction.username == username
    ).all()
    return records
