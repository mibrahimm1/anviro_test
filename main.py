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

def generate_tags_with_groq(text: str, entities: List[str], max_tags: int = 5) -> List[str]:
    """Generate meaningful contextual tags using Groq (LLaMA3)."""
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY not configured")

    entities_list = ", ".join(entities) if entities else "None"

    prompt = f"""
You are an intelligent tag generator.

Given the text below, create {max_tags} short descriptive tags (each 1–3 words)
that summarize the *context, purpose, or theme* of the text.

DO NOT repeat any of these entity names:
{entities_list}

Focus on intent or activity — for example:
- If the text describes travel or meetings, tags could include "business trip", "meeting", "schedule".
- If it describes a product launch, tags could include "product launch", "marketing", "announcement".

Return ONLY a valid JSON array of lowercase strings.

Example:
["business trip", "meeting", "schedule"]

Text:
{text}
"""

    response = groq_client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[{"role": "user", "content": prompt.strip()}],
        temperature=0.4,
        max_tokens=120,
    )

    content = response.choices[0].message["content"].strip()

    import json, re
    try:
        # clean stray markdown or text
        content = re.sub(r"^```(?:json)?|```$", "", content).strip()
        tags = json.loads(content)
        if isinstance(tags, list) and all(isinstance(t, str) for t in tags):
            return [t.strip() for t in tags][:max_tags]
        else:
            raise ValueError("Response not a valid JSON list of strings")
    except Exception:
        # Only fallback if Groq completely fails
        return ["context extraction failed"]


@app.post("/extract", response_model=ExtractOut)
async def extract_endpoint(payload: TextIn):
    text = payload.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="text must be non-empty")

    entities = extract_entities(text)
    try:
        tags = generate_tags_with_groq(text, entities)
    except Exception as e:
        logging.exception("Groq tag generation failed, returning fallback tags")
        tags = entities[:5]  # fallback
    return {"entities": entities, "tags": tags, "meta": {"len_text": len(text)}}

@app.get("/")
def root():
    return {
        "message": "Smart Entity Extraction Microservice - Ibrahim Shaikh",
        "usage": "POST /extract with JSON { 'text': '...' }"
    }