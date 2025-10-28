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
print("GROQ_API_KEY:", GROQ_API_KEY)

groq_client = Groq(api_key=GROQ_API_KEY)

try:
    nlp = spacy.load("en_core_web_sm")
except Exception as e:
    raise RuntimeError(
        "spaCy model 'en_core_web_sm' not found. Install it with: python -m spacy download en_core_web_sm"
    ) from e

app = FastAPI(title="Smart Entity Extraction Microservice")

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
You are a professional assistant that summarizes the *context* of a text using short descriptive tags.

## TASK:
Given a piece of text, generate {max_tags} concise tags (1–3 words each) that describe the *purpose, action, or situation* — not the named entities.

## DO NOT:
- Repeat or include these entities: {entities_list}
- Mention people's names, organizations, or locations.

## DO:
- Focus on the meaning or event (e.g., travel, meetings, deals, reports, etc.)
- Return only a JSON array of lowercase strings.

## EXAMPLES

Input:
"John Doe from Acme Corp traveled to New York on July 20 for a business meeting."
Output:
["business trip", "meeting", "schedule"]

---

Input:
"Apple unveiled the new iPhone 16 during their launch event in California."
Output:
["product launch", "technology", "marketing"]

---

Input:
"Sarah joined Microsoft as a software engineer."
Output:
["career move", "hiring", "tech industry"]

---

Now analyze this text and produce only the JSON array of {max_tags} tags:

{text}
"""
    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt.strip()}],
        temperature=0.4,
        max_tokens=150,
    )

    import json, re

    content = response.choices[0].message.content.strip()

    # Clean and parse JSON safely
    content = re.sub(r"^```(?:json)?|```$", "", content).strip()
    try:
        tags = json.loads(content)
        if isinstance(tags, list) and all(isinstance(t, str) for t in tags):
            # remove duplicates, lowercase
            tags = list(dict.fromkeys([t.lower().strip() for t in tags]))
            return tags[:max_tags]
        else:
            raise ValueError("Invalid JSON response")
    except Exception:
        return ["tag extraction failed"]


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
        tags = entities[:5]
    return {"entities": entities, "tags": tags, "meta": {"len_text": len(text)}}

@app.get("/")
def root():
    return {
        "message": "Smart Entity Extraction Microservice - Ibrahim Shaikh",
        "usage": "POST /extract with JSON { 'text': '...' }"
    }