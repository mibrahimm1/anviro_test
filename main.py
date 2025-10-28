import os
import logging
from typing import List, Dict, Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import spacy
import openai
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

groq_client = Groq(api_key=GROQ_API_KEY)

try:
    nlp = spacy.load("en_core_web_sm")
except Exception as e:
    raise RuntimeError(
        "spaCy model 'en_core_web_sm' not found. Install it with: python -m spacy download en_core_web_sm"
    ) from e

app = FastAPI(title="Smart Entity Extraction Microservice (Groq-powered)")

class TextIn(BaseModel):
    text: str

class ExtractOut(BaseModel):
    entities: List[str]
    tags: List[str]
    meta: Dict[str, Any] = {}

def extract_entities(text: str) -> List[str]:
    """Extract named entities."""
    doc = nlp(text)
    wanted = {"PERSON", "ORG", "GPE", "LOC", "DATE", "EVENT", "PRODUCT", "NORP"}
    ent_vals = [ent.text.strip() for ent in doc.ents if ent.label_ in wanted]
    # dedupe preserving order
    seen = set()
    deduped = []
    for e in ent_vals:
        if e not in seen:
            deduped.append(e)
            seen.add(e)
    return deduped

def generate_tags_with_groq(text: str, max_tags: int = 5) -> List[str]:
    """Generate tags using Groq API (LLaMA3)."""
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY not configured")

    prompt = (
        "You are a concise tag generator. Given the text below, return a JSON array "
        f"of 3 to {max_tags} short tags (each 1–3 words) summarizing key topics or entities. "
        "Respond ONLY with the JSON array and nothing else.\n\n"
        f"Text:\n{text}\n\n"
        "Example output:\n"
        '["tag1","tag2","tag3"]'
    )

    # Call Chat
    response = groq_client.chat.completions.create(
        model="llama3-8b-8192",  # available Groq model
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=150,
    )

    content = response.choices[0].message["content"].strip()

    # Parse JSON
    import json
    try:
        tags = json.loads(content)
        if isinstance(tags, list):
            return [str(t).strip() for t in tags][:max_tags]
        else:
            raise ValueError("Invalid JSON format")
    except Exception:
        # fallback if not valid JSON
        fallback = content.strip("[]").replace("\n", ",")
        parts = [p.strip().strip('"').strip("'") for p in fallback.split(",") if p.strip()]
        return parts[:max_tags]

@app.post("/extract", response_model=ExtractOut)
async def extract_endpoint(payload: TextIn):
    text = payload.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="text must be non-empty")

    entities = extract_entities(text)
    try:
        tags = generate_tags_with_groq(text)
    except Exception as e:
        logging.exception("Groq tag generation failed, returning fallback tags")
        tags = entities[:5]  # fallback
    return {"entities": entities, "tags": tags, "meta": {"len_text": len(text)}}

@app.get("/")
def root():
    return {
        "message": "Smart Entity Extraction Microservice (Groq-powered)",
        "usage": "POST /extract with JSON { 'text': '...' }"
    }