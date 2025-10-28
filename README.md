# Smart Entity Extraction Microservice

A small FastAPI microservice that:
- Extracts named entities (people, orgs, locations, dates) using spaCy `en_core_web_sm`.
- Generates 3–5 short tags using OpenAI Chat Completions API.
- Returns JSON: `{ "entities": [...], "tags": [...] }`

## Quick start (locally)
1. Clone the repo:
   ```bash
   git clone https://github.com/<your-username>/smart-entity-extractor.git
   cd smart-entity-extractor
