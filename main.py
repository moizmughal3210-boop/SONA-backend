from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import base64
import httpx
import os
import json
import tempfile
import requests

load_dotenv()

async def transcribe_with_gemini(
  audio_path: str, 
  mime_type: str
) -> str:
  
  gemini_key = os.getenv("GEMINI_API_KEY")
  
  # Read and encode audio file
  with open(audio_path, "rb") as f:
    audio_data = base64.b64encode(
      f.read()
    ).decode("utf-8")
  
  url = (
    "https://generativelanguage.googleapis.com"
    "/v1beta/models/gemini-2.5-flash"
    f":generateContent?key={gemini_key}"
  )
  
  payload = {
    "contents": [{
      "parts": [
        {
          "inline_data": {
            "mime_type": mime_type,
            "data": audio_data
          }
        },
        {
          "text": (
            "Transcribe this audio exactly "
            "word for word. Return only the "
            "transcription, nothing else."
          )
        }
      ]
    }]
  }
  
  async with httpx.AsyncClient(
    timeout=60.0
  ) as client:
    response = await client.post(
      url, 
      json=payload
    )
    result = response.json()
    
    if "error" in result:
      raise Exception(
        f"Gemini error: {result['error']}"
      )

    print("Gemini raw response:", 
      json.dumps(result, indent=2)[:500])

    if "candidates" not in result:
      raise Exception(
        f"Gemini API error: {result}"
      )
    
    candidate = result["candidates"][0]
    content = candidate["content"]
    parts = content["parts"]

    # Handle both string and list responses
    if isinstance(parts, list):
      transcript = " ".join([
        part.get("text", "") 
        for part in parts
        if isinstance(part, dict)
      ])
    else:
      transcript = str(parts)

    # Clean up the transcript
    transcript = transcript.strip()

    if not transcript:
      raise Exception(
        "Gemini returned empty transcription"
      )

    return transcript

app = FastAPI(
  title="SONA API",
  version="1.0.0"
)

app.add_middleware(
  CORSMiddleware,
  allow_origins=["http://localhost:3000"],
  allow_credentials=True,
  allow_methods=["*"],
  allow_headers=["*"],
)

def run_cf_ai(messages, max_tokens=1000):
    account_id = os.getenv("CLOUDFLARE_ACCOUNT_ID")
    api_token = os.getenv("CLOUDFLARE_API_TOKEN")
    if not account_id or not api_token:
        raise ValueError("Missing CLOUDFLARE_ACCOUNT_ID or CLOUDFLARE_API_TOKEN")
    
    url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/@cf/meta/llama-3.1-8b-instruct"
    headers = {"Authorization": f"Bearer {api_token}"}
    payload = {"messages": messages, "max_tokens": max_tokens}
    
    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    return response.json()["result"]["response"]

@app.get("/")
def root():
  return {
    "app": "SONA",
    "tagline": "Truth Has a Voice. Now So Does Proof.",
    "status": "running"
  }

@app.get("/health")
def health():
  return {"status": "healthy"}

@app.post("/investigate")
async def investigate(
  audio: UploadFile = File(...)
):
  tmp_path = None
  try:
    # Step 1: Save audio temporarily
    suffix = f".{audio.filename.split('.')[-1]}"
    with tempfile.NamedTemporaryFile(
      delete=False,
      suffix=suffix
    ) as tmp:
      content = await audio.read()
      tmp.write(content)
      tmp_path = tmp.name
    
    print(f"Audio saved: {tmp_path}")
    
    # Step 2: Transcribe with Gemini
    print("Transcribing with Gemini...")
    
    mime_map = {
      ".mp3": "audio/mp3",
      ".wav": "audio/wav",
      ".ogg": "audio/ogg",
      ".m4a": "audio/mp4",
      ".webm": "audio/webm"
    }
    mime_type = mime_map.get(
      suffix.lower(), "audio/mp3"
    )

    transcript = await transcribe_with_gemini(
      tmp_path, 
      mime_type
    )
    print(f"Transcript: {transcript[:100]}...")
    
    # Step 3: Extract claims with Claude
    print("Extracting claims with Cloudflare AI...")
    
    claims_text = run_cf_ai(
        max_tokens=1000,
        messages=[{
          "role": "user",
          "content": f"""
          Extract all verifiable factual claims 
          from this transcript.
          Ignore opinions and emotions.
          Only include statements that can be 
          checked against real-world evidence.
          
          Transcript: {transcript}
          
          Return ONLY a valid JSON array:
          [
            {{
              "claim": "exact claim text",
              "timestamp": "approximate position 
                like 0:05-0:10"
            }}
          ]
          
          If no verifiable claims, return: []
          Return ONLY the JSON, no other text.
          """
        }]
    )
    
    print(f"Claims extracted (raw): {type(claims_text)}")
    
    if isinstance(claims_text, list):
        extracted_claims = claims_text
    else:
        if isinstance(claims_text, str):
            claims_text = claims_text.strip()
            # Clean JSON response
            if "```" in claims_text:
                claims_text = claims_text.split("```")[1].replace("json", "").strip()
        print(f"Claims extracted (text): {claims_text}")
        extracted_claims = json.loads(claims_text)
    
    # Step 4: Verify each claim
    print("Verifying claims...")
    verified_claims = []
    
    for claim_obj in extracted_claims[:5]:
      claim = claim_obj["claim"]
      print(f"Verifying: {claim}")
      
      # Search Tavily for evidence
      search_response = requests.post(
        "https://api.tavily.com/search",
        json={
          "api_key": os.getenv("TAVILY_API_KEY"),
          "query": claim,
          "search_depth": "advanced",
          "max_results": 3
        }
      )
      
      search_data = search_response.json()
      search_results = search_data.get(
        "results", []
      )
      
      # Format evidence
      evidence_text = "\n".join([
        f"Source: {r.get('url', '')}\n"
        f"Content: {r.get('content', '')[:300]}"
        for r in search_results
      ])
      
      # Verify with Cloudflare AI
      verdict_text = run_cf_ai(
          max_tokens=500,
          messages=[{
            "role": "user",
            "content": f"""
            Claim: "{claim}"
            
            Evidence:
            {evidence_text if evidence_text 
              else "No evidence found."}
            
            Classify as exactly one of:
            FALSE, MISLEADING, VERIFIED, 
            or UNVERIFIED
            
            Return ONLY this JSON:
            {{
              "verdict": "FALSE|MISLEADING|
                VERIFIED|UNVERIFIED",
              "evidence": "one sentence 
                explanation",
              "confidence": 0-100
            }}
            """
          }]
      )
      
      if isinstance(verdict_text, dict):
          verdict_data = verdict_text
      else:
          if isinstance(verdict_text, str):
              verdict_text = verdict_text.strip()
              if "```" in verdict_text:
                  verdict_text = verdict_text.split("```")[1].replace("json", "").strip()
          verdict_data = json.loads(verdict_text)
      
      verified_claims.append({
        "claim": claim,
        "verdict": verdict_data["verdict"],
        "evidence": verdict_data["evidence"],
        "sources": [
          r.get("url", "")
          for r in search_results[:2]
        ],
        "timestamp": claim_obj.get(
          "timestamp", "Unknown"
        ),
        "confidence": verdict_data["confidence"]
      })
    
    # Cleanup temp file
    if tmp_path and os.path.exists(tmp_path):
      os.unlink(tmp_path)
    
    print("Investigation complete!")
    
    return {
      "filename": audio.filename,
      "transcript": transcript,
      "claims": verified_claims,
      "analyzedAt": "Just now"
    }
    
  except Exception as e:
    print(f"Error: {str(e)}")
    if tmp_path and os.path.exists(tmp_path):
      os.unlink(tmp_path)
    return {
      "error": str(e),
      "filename": getattr(
        audio, 'filename', 'unknown'
      ),
      "transcript": "",
      "claims": [],
      "analyzedAt": "Failed"
    }
