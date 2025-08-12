import os
import json
import tempfile
import requests
from fastapi import FastAPI, Request, UploadFile, File
from google.cloud import speech

app = FastAPI()

# --- Handle Google Cloud credentials from env var on Render ---
if "GOOGLE_CREDENTIALS_JSON" in os.environ:
    creds_json = os.environ["GOOGLE_CREDENTIALS_JSON"]
    with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as f:
        f.write(creds_json.encode())
        temp_json_path = f.name
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = temp_json_path
else:
    raise RuntimeError("Missing GOOGLE_CREDENTIALS_JSON environment variable")

speech_client = speech.SpeechClient()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

@app.post("/parse")
async def parse_command(req: Request):
    data = await req.json()
    text = data.get("text", "")

    prompt = f"""
You are a multilingual cricket commentary interpreter for a live scoring system.
Your job is to deeply understand the cricket event described in the commentary, then output structured match data.

## Reasoning process
1. Translate non-English commentary (Hindi, Marathi, Bengali, Tamil, Hinglish, etc.) to English internally.
2. Understand the actual cricket event, even if phrased casually or as slang.
3. Determine if it’s a scoring shot, wicket, reset, or other event.
4. Break down complex situations into final scoring values (runs, extras, dismissal).

## Special rules for cricket:
- Boundary rules:
  - In air over the rope → 6 runs.
  - After bounce and over rope → 4 runs.
- Extras:
  - Wide ball adds +1 run (plus any runs scored from it).
  - No ball adds +1 run (plus any runs scored from it).
  - Bye and leg bye runs count towards extras but not batsman runs.
- Overthrows:
  - Add overthrow runs to the runs actually taken.
  - Example: "3 runs taken, plus overthrow for 2 more" → total runs = 5.
- Free hits:
  - Recognize "free hit" mention but it does not change scoring logic except no wicket on that ball.
- Penalty runs:
  - If mentioned (e.g., “5 penalty runs”), add directly to total runs.
- Wickets:
  - Recognize "bowled", "caught", "run out", "stumped", "lbw", "hit wicket".
  - If type unclear, set dismissal = null.

## Output only JSON:
{{
  "event": "score" | "wicket" | "reset" | "other",
  "runs": <integer>,
  "extras": "wide" | "no ball" | "bye" | "leg bye" | null,
  "dismissal": "bowled" | "caught" | "run out" | "stumped" | "lbw" | "hit wicket" | null
}}

Example interpretations:
- "Ball goes over the boundary without bouncing" → {{ "event": "score", "runs": 6, "extras": null, "dismissal": null }}
- "Three runs plus an overthrow for two more" → {{ "event": "score", "runs": 5, "extras": null, "dismissal": null }}
- "Batsman is bowled" → {{ "event": "wicket", "runs": 0, "extras": null, "dismissal": "bowled" }}
- "Wide ball and batsmen run two" → {{ "event": "score", "runs": 3, "extras": "wide", "dismissal": null }}
- "No ball hit for four" → {{ "event": "score", "runs": 5, "extras": "no ball", "dismissal": null }}
- "5 penalty runs to batting side" → {{ "event": "score", "runs": 5, "extras": null, "dismissal": null }}

Now, process this commentary: "{text}"
"""

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
        json=payload  # use json= instead of manual dumps
    )

    if groq_response.status_code != 200:
        return {"error": "Groq API error", "details": groq_response.text}

    try:
        content = groq_response.json()["choices"][0]["message"]["content"]
        parsed_json = json.loads(content)
    except json.JSONDecodeError:
        return {"error": "Failed to parse LLM output as JSON", "raw": content}
    except Exception as e:
        return {"error": str(e)}

    return parsed_json

@app.post("/stt")
async def stt_endpoint(request: Request):
    try:
        # Get raw audio bytes
        content = await request.body()
        if not content:
            return {"error": "No audio data received"}

        # If sending WAV file, strip the first 44 bytes (WAV header)
        if content[0:4] == b"RIFF":
            content = content[44:]

        audio = speech.RecognitionAudio(content=content)
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000,
            language_code="en-IN",
            alternative_language_codes=["hi-IN", "mr-IN"],
            enable_automatic_punctuation=True
        )

        response = speech_client.recognize(config=config, audio=audio)
        if not response.results:
            return {"transcript": ""}

        transcript = " ".join(
            [result.alternatives[0].transcript for result in response.results]
        )
        return {"transcript": transcript}

    except Exception as e:
        return {"error": str(e)}
