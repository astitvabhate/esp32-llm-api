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
    You are a cricket match voice command parser.
    Given this input: "{text}", output JSON only with:
    {{
      "event": "score" | "wicket" | "reset" | "other",
      "runs": <integer>,
      "extras": "wide" | "no ball" | null,
      "dismissal": "bowled" | "caught" | "run out" | null
    }}
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
