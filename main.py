import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv(encoding="utf-8")

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY is missing in .env file")

client = genai.Client(api_key=api_key)

app = FastAPI(title="AI Mobile App Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str

@app.get("/")
def home():
    return {"status": "online"}

@app.post("/api/chat")
async def chat(request: ChatRequest):
    try:
        sys_instruction = (
            "You are a helpful AI tutor for students in Kyrgyzstan preparing for national exams (ORT/JORT). "
            "Respond concisely, friendly, and structured for a mobile screen. "
            "Reply in Russian or Kyrgyz depending on the user's language."
        )
        
        response = client.models.generate_content(
            model="models/gemma-4-26b-a4b-it",
            contents=str(request.message),
            config=types.GenerateContentConfig(
                system_instruction=sys_instruction,
                temperature=0.7,
            )
        )
        return {"reply": response.text}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))