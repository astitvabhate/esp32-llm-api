from fastapi import FastAPI, Request
import os
import requests
import json

app = FastAPI()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

@app.post("/parse")
async def parse_command(req: Request):
    data = await req.json()
    text = data.get("text", "")

    # Define the system prompt
    prompt = f"""
You are a multilingual cricket commentary interpreter for a live scoring system.
Your job is to read what the commentator or person says, understand the cricket event described,
and output structured match data.

**Your reasoning process:**
1. If commentary describes a ball going to or over the boundary → decide if it was a 4 (ground shot) or 6 (hit in the air without bounce).
2. If commentary mentions a batsman being dismissed → event = "wicket" and fill dismissal type if known.
3. If commentary mentions runs being taken → event = "score" and set runs accordingly.
4. If extras (wide, no ball, bye, leg bye) are mentioned → set extras field.
5. If it describes a reset, innings change, or restart → event = "reset".
6. If unclear, event = "other".

**Language handling:**
- The commentary may be in English, Hindi, Marathi, Bengali, Tamil, or a mix (Hinglish).
- Translate internally to English before understanding.
- Focus on meaning, not exact words.

**Output only this JSON**:
{{
  "event": "score" | "wicket" | "reset" | "other",
  "runs": <integer>,
  "extras": "wide" | "no ball" | "bye" | "leg bye" | null,
  "dismissal": "bowled" | "caught" | "run out" | "stumped" | "lbw" | null
}}

Now, process this commentary: "{text}"
"""


    # Call Groq API
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "llama-3.1-8b-instant",
        "messages": [
            {"role": "system", "content": "You are a cricket voice command parser."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0
    }

    groq_response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers=headers,
        data=json.dumps(payload)
    )

    if groq_response.status_code != 200:
        return {"error": "Groq API error", "details": groq_response.text}

    # Extract the JSON from the model's output
    try:
        content = groq_response.json()["choices"][0]["message"]["content"]
        parsed_json = json.loads(content)
    except Exception:
        return {"error": "Failed to parse LLM output", "raw": content}

    return parsed_json
