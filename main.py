from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import base64
import httpx
import os
import json
import tempfile
import requests
import sqlite3
import hashlib
import time
import google.generativeai as genai

load_dotenv()

async def transcribe_with_groq(
  audio_path: str, 
  mime_type: str,
  language: str = "auto"
) -> str:
  
  groq_key = os.getenv("GROQ_API_KEY")
  if not groq_key:
    raise ValueError("Missing GROQ_API_KEY in .env")
    
  url = "https://api.groq.com/openai/v1/audio/transcriptions"
  headers = {
    "Authorization": f"Bearer {groq_key}"
  }
  
  with open(audio_path, "rb") as f:
    files = {
      "file": (os.path.basename(audio_path), f, mime_type)
    }
    data = {
      "model": "whisper-large-v3-turbo",
      "prompt": "Transcribe this audio accurately. It may contain Urdu, English, mixed Urdu-English (code-switching), Roman Urdu, or informal slang. Keep the original words and do not translate. Preserve all language mixing exactly.",
      "response_format": "json"
    }

    # Pass language parameter if specified and not mixed/auto
    if language == "ur":
        data["language"] = "ur"
    elif language == "en":
        data["language"] = "en"
    
    async with httpx.AsyncClient(timeout=60.0) as client:
      response = await client.post(url, headers=headers, data=data, files=files)
      
      if response.status_code != 200:
        raise Exception(f"Groq API error: {response.text}")
        
      result = response.json()
      transcript = result.get("text", "").strip()
      
      if not transcript:
        raise Exception("Groq returned empty transcription")
        
      return transcript

# Database setup for Feature 9 (Memory) and Feature 8 (Evolution)
DB_PATH = "sona_memory.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS claims (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            claim TEXT,
            verdict TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS threat_patterns (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pattern_key TEXT UNIQUE,
            category TEXT,
            description TEXT,
            indicators TEXT,
            first_seen DATETIME DEFAULT CURRENT_TIMESTAMP,
            last_seen DATETIME DEFAULT CURRENT_TIMESTAMP,
            occurrence_count INTEGER DEFAULT 1,
            status TEXT DEFAULT 'EMERGING'
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS community_contributions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            investigation_hash TEXT UNIQUE,
            pattern_key TEXT
        )
    ''')
    conn.commit()
    conn.close()

init_db()

def find_similar_claims(new_claim: str):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    # Simple substring matching for MVP Memory/Evolution
    words = [w for w in new_claim.split() if len(w) > 4][:3]
    if not words:
        return []
    
    query = "SELECT filename, claim, verdict, timestamp FROM claims WHERE " + " OR ".join(["claim LIKE ?" for _ in words])
    cursor.execute(query, [f"%{w}%" for w in words])
    results = cursor.fetchall()
    conn.close()
    
    matches = []
    for r in results:
        matches.append({
            "previous_filename": r[0],
            "previous_claim": r[1],
            "previous_verdict": r[2],
            "date": r[3]
        })
    return matches

def save_claim_to_db(filename: str, claim: str, verdict: str):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO claims (filename, claim, verdict) VALUES (?, ?, ?)", (filename, claim, verdict))
    conn.commit()
    conn.close()

# Feature 7: Audio Integrity Lab
def analyze_audio_integrity(audio_path: str):
    # This is a lightweight proxy for actual waveform analysis.
    # In a full production environment, librosa or wave would be used here.
    file_size = os.path.getsize(audio_path)
    events = []
    if file_size < 50000:
        events.append({
            "type": "Low Bitrate/Compression",
            "time": "Entire file",
            "signal": "Possible high compression detected",
            "explanation": "Highly compressed audio can mask edits.",
            "severity": "Low"
        })
    return events

app = FastAPI(title="SONA API", version="1.0.0")

app.add_middleware(
  CORSMiddleware,
  allow_origins=["http://localhost:3000"],
  allow_credentials=True,
  allow_methods=["*"],
  allow_headers=["*"],
)

def run_cf_ai(messages, max_tokens=2000):
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
    "status": "running"
  }

@app.get("/health")
def health():
  return {"status": "healthy"}

from fastapi import Form

class UnifiedRequest(BaseModel):
    input_type: str # "text", "url", "message", "audio"
    content: str

def analyze_language_scam_intelligence(text: str):
    prompt = f"""
    Analyze the following text for language, social engineering signals, and Pakistan-focused scam indicators.
    Understand Urdu, Roman Urdu, English, and code-switching.
    Text: {text}
    
    Return ONLY a JSON object exactly matching this schema:
    {{
      "detected_language": "e.g., Roman Urdu + English",
      "script": "e.g., Latin / Code-switching",
      "is_code_switching": true,
      "normalized_text": "Meaning normalized to English",
      "social_engineering_signals": [
         {{"signal": "URGENCY", "evidence_text": "exact words", "confidence": 90}}
      ],
      "scam_patterns": ["Delivery / Parcel Scam", "OTP Request"],
      "risk_level": "HIGH",
      "explanation": "Why SONA flagged this. Be precise based on context.",
      "risk_indicators": [
         {{"indicator": "Urgency/Pressure", "confidence": 90, "description": "..."}}
      ],
      "recommended_actions": [
         "Do not share OTP"
      ]
    }}
    """
    res = run_cf_ai([{"role": "user", "content": prompt}], max_tokens=1500)
    try:
        if isinstance(res, dict): return res
        if "```" in res: res = res.split("```")[1].replace("json","").strip()
        return json.loads(res)
    except:
        return {"risk_indicators": [], "recommended_actions": []}

def reconstruct_attack(transcript: str, verified_claims: list, risk_analysis: dict):
    prompt = f"""
    You are SONA's Attack Reconstruction Engine. Based on the provided text, verified claims, and risk indicators, construct a possible attack, manipulation, scam, or misinformation timeline and assessment.
    
    IMPORTANT RULES:
    1. Do NOT state inferred intent or future actions as confirmed facts. Use words like "Possible", "Potential", or "Likely".
    2. If there is no evidence of an attack, set attack_type to "No Attack Detected" and keep timeline minimal.
    3. Do NOT invent timestamps. Use "Time unavailable".
    4. Only output a valid JSON object matching this schema exactly.
    
    Text: {transcript}
    
    Risk Indicators: {json.dumps(risk_analysis.get('risk_indicators', []))}
    Verified Claims: {json.dumps([c['claim'] for c in verified_claims])}
    
    Return ONLY JSON:
    {{
      "attack_summary": "A short summary of what likely happened",
      "attack_type": "Potential Social Engineering Attack",
      "attack_confidence": 85,
      "timeline": [
        {{
          "event_id": "evt_1",
          "sequence_number": 1,
          "title": "Initial Contact",
          "description": "Explanation of the event",
          "event_type": "Contact",
          "timestamp": "Time unavailable",
          "confidence": "HIGH",
          "evidence": "Caller stated a parcel arrived"
        }}
      ],
      "techniques": ["Impersonation", "Urgency", "OTP Request"],
      "predicted_next_steps": [
        {{"step": "Attempt to gain access to account", "confidence": "HIGH"}}
      ],
      "impact_assessment": {{
        "Account Security": {{"level": "HIGH", "explanation": "OTP requested"}},
        "Financial Risk": {{"level": "LOW", "explanation": "No payment requested"}},
        "Privacy Risk": {{"level": "MEDIUM", "explanation": "Name/phone known"}},
        "Contact Risk": {{"level": "UNKNOWN", "explanation": ""}}
      }},
      "recommended_actions": [
        "Do not share OTP",
        "Verify delivery directly with official app"
      ]
    }}
    """
    res = run_cf_ai([{"role": "user", "content": prompt}], max_tokens=2000)
    try:
        if isinstance(res, dict): return res
        if "```" in res: res = res.split("```")[1].replace("json","").strip()
        return json.loads(res)
    except:
        return {}

def process_community_intelligence(transcript: str, risk_analysis: dict, investigation_hash: str):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM community_contributions WHERE investigation_hash = ?", (investigation_hash,))
    if cursor.fetchone():
        conn.close()
        return None
        
    prompt = f"""
    You are SONA's Privacy Filter & Pattern clustering engine.
    Extract normalized threat indicators from this investigation.
    DO NOT include PII, names, phone numbers, OTPs, or exact messages.
    
    Transcript: {transcript}
    Scam Intelligence: {json.dumps(risk_analysis)}
    
    If this is not a scam/threat, return {{"is_threat": false}}.
    If it is a threat, return JSON:
    {{
       "is_threat": true,
       "pattern_key": "DELIVERY_OTP_SCAM",
       "category": "Delivery Scam",
       "description": "Caller claims parcel delivery and requests verification code",
       "indicators": ["Delivery impersonation", "OTP request", "Urgency"]
    }}
    """
    res = run_cf_ai([{"role": "user", "content": prompt}], max_tokens=1000)
    try:
        if isinstance(res, dict): data = res
        elif "```" in res: data = json.loads(res.split("```")[1].replace("json","").strip())
        else: data = json.loads(res)
        
        if data.get("is_threat") and data.get("pattern_key"):
            pattern_key = data["pattern_key"]
            category = data.get("category", "")
            description = data.get("description", "")
            indicators = json.dumps(data.get("indicators", []))
            
            cursor.execute("SELECT id, occurrence_count FROM threat_patterns WHERE pattern_key = ?", (pattern_key,))
            row = cursor.fetchone()
            if row:
                new_count = row[1] + 1
                status = 'ESTABLISHED' if new_count >= 3 else 'OBSERVED'
                cursor.execute("UPDATE threat_patterns SET occurrence_count = ?, last_seen = CURRENT_TIMESTAMP, status = ? WHERE pattern_key = ?", (new_count, status, pattern_key))
            else:
                cursor.execute("INSERT INTO threat_patterns (pattern_key, category, description, indicators, status) VALUES (?, ?, ?, ?, 'EMERGING')", (pattern_key, category, description, indicators))
            
            cursor.execute("INSERT INTO community_contributions (investigation_hash, pattern_key) VALUES (?, ?)", (investigation_hash, pattern_key))
            conn.commit()
    except Exception as e:
        print("Community intelligence error:", e)
    finally:
        conn.close()

def get_threat_pattern_match(transcript: str, risk_analysis: dict):
    prompt = f"""
    Analyze if this investigation matches any known abstract scam pattern (e.g. DELIVERY_OTP_SCAM, BANK_IMPERSONATION).
    Transcript: {transcript}
    Scam Intelligence: {json.dumps(risk_analysis)}
    Return ONLY JSON:
    {{
       "pattern_key": "DELIVERY_OTP_SCAM"
    }}
    Or pattern_key: null if no clear pattern.
    """
    res = run_cf_ai([{"role": "user", "content": prompt}], max_tokens=500)
    try:
        if isinstance(res, dict): data = res
        elif "```" in res: data = json.loads(res.split("```")[1].replace("json","").strip())
        else: data = json.loads(res)
        
        pattern_key = data.get("pattern_key")
        if not pattern_key: return None
        
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT category, description, indicators, occurrence_count, status FROM threat_patterns WHERE pattern_key = ?", (pattern_key,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return {
                "pattern_key": pattern_key,
                "category": row[0],
                "description": row[1],
                "indicators": json.loads(row[2]) if row[2] else [],
                "confidence": min(50 + (row[3] * 10), 98),
                "status": row[4],
                "occurrences": row[3]
            }
        return None
    except:
        return None

def analyze_incident_state(transcript: str, language_intelligence: dict, user_answers: dict = None):
    prompt = f"""
    You are SONA's Incident Response & Recovery Engine.
    Determine the incident state, prevention guidance, checkpoint questions, and recovery steps.
    
    Transcript: {transcript}
    Scam Intelligence: {json.dumps(language_intelligence)}
    User Answers: {json.dumps(user_answers or dict())}
    
    States: NO_RISK, POTENTIAL_RISK, HIGH_RISK, POSSIBLE_COMPROMISE, CONFIRMED_BY_USER, RECOVERY_IN_PROGRESS, CONTAINED, RESOLVED, UNKNOWN
    
    Rules:
    1. If Scam Intelligence shows risk (e.g. OTP request) but User Answers are empty, state is POTENTIAL_RISK. prevention_mode=true. Provide user_checkpoint_questions (e.g. "Have you shared the code?").
    2. If User Answers indicate risky action taken (e.g. "Yes"), state is POSSIBLE_COMPROMISE. prevention_mode=false. Provide immediate_actions, recovery_checklist, contact_warning_message, post_incident_risk.
    3. Generate contact_warning_message in the same language as the transcript.
    
    Return ONLY JSON:
    {{
      "incident_state": "POTENTIAL_RISK",
      "prevention_mode": true,
      "user_checkpoint_questions": [
        {{
          "id": "shared_code",
          "question": "Have you shared the verification code?",
          "options": ["No", "Yes", "I'm not sure"]
        }}
      ],
      "prevention_guidance": [
        "Do NOT share the code.",
        "End the conversation."
      ],
      "immediate_actions": [
        {{"priority": "DO THIS NOW", "action": "Secure account"}}
      ],
      "recovery_checklist": [
        {{"id": "rec_1", "step": "Check active sessions", "completed": false}}
      ],
      "contact_warning_message": "⚠️ Important: My account may have been compromised...",
      "post_incident_risk": {{
         "Account Security": {{"level": "HIGH", "explanation": "..."}},
         "Financial Risk": {{"level": "MEDIUM", "explanation": "..."}}
      }}
    }}
    """
    res = run_cf_ai([{"role": "user", "content": prompt}], max_tokens=2000)
    try:
        if isinstance(res, dict): return res
        if "```" in res: res = res.split("```")[1].replace("json","").strip()
        return json.loads(res)
    except:
        return {
            "incident_state": "UNKNOWN", "prevention_mode": False,
            "user_checkpoint_questions": [], "prevention_guidance": [],
            "contact_warning_message": "", "post_incident_risk": {}
        }

def analyze_case_intelligence(transcript: str, risk_analysis: dict, attack_reconstruction: dict, verified_claims: list):
    prompt = f"""
    You are SONA's AI Investigation Copilot.
    Analyze the case and return a structured JSON response identifying the strongest evidence, gaps, contradictions, and next actions.

    Transcript: {transcript}
    Scam Intelligence: {json.dumps(risk_analysis)}
    Attack Reconstruction: {json.dumps(attack_reconstruction)}
    Verified Claims: {json.dumps([c.get("claim") for c in verified_claims])}

    Return ONLY JSON:
    {{
      "current_assessment": "Likely social-engineering...",
      "investigation_status": "NEEDS_VERIFICATION",
      "risk_level": "HIGH",
      "confidence": 82,
      "strongest_evidence": ["Caller claimed to represent courier", "Requested verification code"],
      "missing_information": ["Official courier confirmation", "Caller identity not independently verified"],
      "contradictions": ["No official tracking record found"],
      "next_best_actions": [
        {{"action": "Verify Courier Information", "reason": "Independent verification can significantly change the verdict", "priority": "HIGH"}}
      ],
      "smart_questions": [
        "Were you expecting a parcel?",
        "Did you share the verification code?"
      ],
      "what_would_change_verdict": "If the tracking number is verified through official channels, this could be legitimate."
    }}
    """
    res = run_cf_ai([{"role": "user", "content": prompt}], max_tokens=2500)
    try:
        if isinstance(res, dict): return res
        if "```" in res: res = res.split("```")[1].replace("json","").strip()
        return json.loads(res)
    except:
        return {}

async def run_investigation_pipeline(transcript: str, filename: str, language: str = "English", audio_integrity_events=None):
    word_count = len(transcript.split())
    claims_text = run_cf_ai(
        max_tokens=2000,
        messages=[{
          "role": "user",
          "content": f"""
          Extract all verifiable factual claims from this text.
          Ignore opinions and emotions.
          
          Text: {transcript}
          
          Return ONLY a valid JSON array of objects:
          [
            {{
              "claim": "normalized English version",
              "original_claim": "exact words",
              "language": "Urdu+English",
              "timestamp_start": "00:00",
              "timestamp_end": "00:00",
              "transcript_segment": "exact text here"
            }}
          ]
          If no verifiable claims, return: []
          Return ONLY the JSON, no other text.
          """
        }]
    )
    
    if isinstance(claims_text, list):
        extracted_claims = claims_text
    else:
        if isinstance(claims_text, str):
            claims_text = claims_text.strip()
            if "```" in claims_text:
                claims_text = claims_text.split("```")[1].replace("json", "").strip()
        extracted_claims = json.loads(claims_text)
    
    verified_claims = []
    total_sources = 0
    if extracted_claims and "language" in extracted_claims[0]:
        language = extracted_claims[0]["language"]
    
    overall_claim_accuracy = 0
    overall_context_completeness = 0
    overall_source_quality = 0
    overall_evidence_consistency = 0

    for claim_obj in extracted_claims[:5]:
      claim = claim_obj["claim"]
      search_response = requests.post(
        "https://api.tavily.com/search",
        json={"api_key": os.getenv("TAVILY_API_KEY"),"query": claim,"search_depth": "advanced","max_results": 3}
      )
      search_data = search_response.json()
      search_results = search_data.get("results", [])
      
      pk_search_response = requests.post(
        "https://api.tavily.com/search",
        json={
          "api_key": os.getenv("TAVILY_API_KEY"),
          "query": f"{claim} site:dawn.com OR site:geo.tv OR site:thenews.com.pk OR site:tribune.com.pk OR site:pbs.gov.pk OR site:sbp.org.pk OR site:pta.gov.pk OR site:fia.gov.pk",
          "search_depth": "basic",
          "max_results": 2
        }
      )
      pk_results = pk_search_response.json().get("results", [])
      pk_sources = [{
        **r,
        "is_pakistani_source": True,
        "source_flag": "🇵🇰"
      } for r in pk_results]
      
      all_sources = search_results + pk_sources
      total_sources += len(all_sources)
      evidence_text = "\\n".join([f"Source: {r.get('url', '')}\\nContent: {r.get('content', '')[:300]}" for r in search_results])
      
      if pk_sources:
          evidence_text += "\\n\\nPakistani Sources:\\n"
          evidence_text += "\\n".join([
            f"Source: {r.get('url','')}\\n{r.get('content','')[:200]}"
            for r in pk_sources
          ])
      
      verdict_text = run_cf_ai(
          max_tokens=2500,
          messages=[{"role": "user", "content": f"""
            Claim: "{claim}"
            Evidence: {evidence_text if evidence_text else "No evidence found."}
            Return ONLY JSON:
            {{
              "verdict": "FALSE" | "MISLEADING" | "VERIFIED" | "UNVERIFIED",
              "evidence": "explanation",
              "confidence": 92,
              "supporting_sources": [{{"url": "...", "snippet": "..."}}],
              "contradicting_sources": [{{"url": "...", "snippet": "..."}}],
              "context_sources": [{{"url": "...", "snippet": "..."}}],
              "conflict_status": "Evidence Agrees|Evidence Conflicts|Mixed Evidence|Insufficient Evidence",
              "suggestions": [],
              "what_would_change": ["what missing evidence would change this verdict"],
              "context_status": "Context Intact" | "Missing Context" | "Outdated" | "Inconclusive",
              "context_summary": "...",
              "missing_context": ["..."],
              "original_source_candidates": ["..."],
              "evidence_strength": {{
                  "source_quality": {{"value": "High|Medium|Low|Insufficient", "explanation": "..."}},
                  "source_independence": {{"value": "High|Medium|Low|Insufficient", "explanation": "..."}},
                  "evidence_agreement": {{"value": "High|Medium|Low|Insufficient", "explanation": "..."}},
                  "evidence_recency": {{"value": "High|Medium|Low|Insufficient", "explanation": "..."}},
                  "directness_of_evidence": {{"value": "High|Medium|Low|Insufficient", "explanation": "..."}},
                  "context_completeness": {{"value": "High|Medium|Low|Insufficient", "explanation": "..."}}
              }}
            }}
            """}]
      )
      
      if isinstance(verdict_text, dict):
          verdict_data = verdict_text
      else:
          if isinstance(verdict_text, str):
              verdict_text = verdict_text.strip()
              if "```" in verdict_text:
                  verdict_text = verdict_text.split("```")[1].replace("json", "").strip()
          verdict_data = json.loads(verdict_text)
      
      if verdict_data.get("verdict") == "VERIFIED": overall_claim_accuracy += 100
      elif verdict_data.get("verdict") == "MISLEADING": overall_claim_accuracy += 50
      
      strength = verdict_data.get("evidence_strength", {})
      if strength.get("context_completeness", {}).get("value") == "High": overall_context_completeness += 100
      elif strength.get("context_completeness", {}).get("value") == "Medium": overall_context_completeness += 50
      
      if strength.get("source_quality", {}).get("value") == "High": overall_source_quality += 100
      elif strength.get("source_quality", {}).get("value") == "Medium": overall_source_quality += 50
      
      if strength.get("evidence_agreement", {}).get("value") == "High": overall_evidence_consistency += 100
      elif strength.get("evidence_agreement", {}).get("value") == "Medium": overall_evidence_consistency += 50

      save_claim_to_db(filename, claim, verdict_data.get("verdict", "UNVERIFIED"))
      historical_matches = find_similar_claims(claim)
      
      verified_claims.append({
        "claim": claim,
        "original_claim": claim_obj.get("original_claim", claim),
        "language": claim_obj.get("language", "English"),
        "timestamp_start": claim_obj.get("timestamp_start", "00:00"),
        "timestamp_end": claim_obj.get("timestamp_end", "00:00"),
        "transcript_segment": claim_obj.get("transcript_segment", ""),
        "verdict": verdict_data.get("verdict", "UNVERIFIED"),
        "evidence": verdict_data.get("evidence", ""),
        "confidence": verdict_data.get("confidence", 0),
        "supporting_sources": verdict_data.get("supporting_sources", []),
        "contradicting_sources": verdict_data.get("contradicting_sources", []),
        "context_sources": verdict_data.get("context_sources", []),
        "conflict_status": verdict_data.get("conflict_status", "Insufficient Evidence"),
        "suggestions": verdict_data.get("suggestions", []),
        "what_would_change": verdict_data.get("what_would_change", []),
        "context_status": verdict_data.get("context_status", "Inconclusive"),
        "context_summary": verdict_data.get("context_summary", ""),
        "missing_context": verdict_data.get("missing_context", []),
        "original_source_candidates": verdict_data.get("original_source_candidates", []),
        "evidence_strength": verdict_data.get("evidence_strength", {}),
        "historical_matches": historical_matches,
        "pakistani_sources": [r.get("url","") for r in pk_sources],
        "has_pakistani_source": len(pk_sources) > 0
      })
    
    num_claims = len(verified_claims) or 1
    risk_analysis = analyze_language_scam_intelligence(transcript)
    attack_reconstruction = reconstruct_attack(transcript, verified_claims, risk_analysis)
    incident_response = analyze_incident_state(transcript, risk_analysis, None)
    community_match = get_threat_pattern_match(transcript, risk_analysis)
    case_intelligence = analyze_case_intelligence(transcript, risk_analysis, attack_reconstruction, verified_claims)
    
    return {
      "investigation_id": hashlib.md5(transcript.encode()).hexdigest(),
      "filename": filename,
      "transcript": transcript,
      "language": language,
      "language_confidence": 94,
      "word_count": word_count,
      "source_count": total_sources,
      "analyzedAt": "Just now",
      "claims": verified_claims,
      "profile": {
          "claim_accuracy": overall_claim_accuracy // num_claims,
          "context_completeness": overall_context_completeness // num_claims,
          "source_quality": overall_source_quality // num_claims,
          "evidence_consistency": overall_evidence_consistency // num_claims
      },
      "audio_integrity": audio_integrity_events or [],
      "risk_indicators": risk_analysis.get("risk_indicators", []),
      "recommended_actions": risk_analysis.get("recommended_actions", []),
      "language_intelligence": risk_analysis,
      "attack_reconstruction": attack_reconstruction,
      "incident_response": incident_response,
      "community_threat_match": community_match,
      "copilot": case_intelligence
    }

@app.post("/investigate/unified")
async def investigate_unified(req: UnifiedRequest):
    try:
        content = req.content
        if req.input_type == "url":
            import urllib.request
            # Basic safe metadata extraction proxy for URL
            content = f"The user submitted URL: {req.content} for analysis."
            
        return await run_investigation_pipeline(content, f"{req.input_type}_input")
    except Exception as e:
        return {"error": str(e)}

class IncidentUpdateRequest(BaseModel):
    transcript: str
    language_intelligence: dict
    user_answers: dict

@app.post("/investigate/incident-update")
async def update_incident(req: IncidentUpdateRequest):
    try:
        response = analyze_incident_state(req.transcript, req.language_intelligence, req.user_answers)
        return response
    except Exception as e:
        return {"error": str(e)}

class ContributeRequest(BaseModel):
    investigation_id: str
    transcript: str
    language_intelligence: dict

@app.post("/investigate/community-contribute")
async def community_contribute(req: ContributeRequest):
    process_community_intelligence(req.transcript, req.language_intelligence, req.investigation_id)
    return {"status": "success"}

@app.post("/investigate")
async def investigate(audio: UploadFile = File(...), language: str = Form("auto")):
  tmp_path = None
  try:
    suffix = f".{audio.filename.split('.')[-1]}"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
      content = await audio.read()
      tmp.write(content)
      tmp_path = tmp.name
    
    mime_map = {
      ".mp3": "audio/mp3",
      ".wav": "audio/wav",
      ".ogg": "audio/ogg",
      ".m4a": "audio/mp4",
      ".webm": "audio/webm"
    }
    mime_type = mime_map.get(suffix.lower(), "audio/mp3")

    audio_integrity_events = analyze_audio_integrity(tmp_path)

    transcript = await transcribe_with_groq(tmp_path, mime_type, language)
    return await run_investigation_pipeline(transcript, audio.filename, language, audio_integrity_events)
    
  except Exception as e:
    print(f"Error: {str(e)}")
    if tmp_path and os.path.exists(tmp_path):
      os.unlink(tmp_path)
    return {
      "error": str(e),
      "filename": getattr(audio, 'filename', 'unknown'),
      "transcript": "",
      "claims": [],
      "analyzedAt": "Failed"
    }

class AskRequest(BaseModel):
    query: str
    report: dict

def analyze_media_with_gemini(file_path: str, mime_type: str, mode: str):
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    uploaded_file = genai.upload_file(file_path, mime_type=mime_type)
    
    while uploaded_file.state.name == "PROCESSING":
        time.sleep(2)
        uploaded_file = genai.get_file(uploaded_file.name)
        
    model = genai.GenerativeModel("gemini-1.5-pro")
    
    prompt = f"""
    You are SONA's Multi-Modal Forensics Engine.
    Mode: {mode}
    
    Analyze this media file for:
    1. Visible text (OCR). YOU MUST REDACT sensitive info like OTP, CNIC, passwords, bank/card numbers, private phones with [REDACTED].
    2. Image/Video Forensics (metadata clues if visual, visible manipulations, inconsistencies, deepfake indicators).
    3. If Screenshot: detect fake payment confirmations, courier scams, inconsistent UI.
    4. If Audio/Video: check A/V consistency.
    
    Return ONLY JSON:
    {{
      "ocr_text": "Extracted text here...",
      "manipulation_assessment": "Short summary of findings",
      "confidence": 75,
      "severity": "HIGH|MEDIUM|LOW|UNKNOWN",
      "status": "LIKELY_MANIPULATED|POSSIBLY_MANIPULATED|NO_CLEAR_MANIPULATION|INCONCLUSIVE",
      "evidence": ["list", "of", "clues"],
      "limitations": "What cannot be determined"
    }}
    """
    
    try:
        response = model.generate_content([prompt, uploaded_file])
        res_text = response.text.strip()
        if "```" in res_text:
            res_text = res_text.split("```")[1].replace("json", "").strip()
        data = json.loads(res_text)
    except Exception as e:
        data = {
            "ocr_text": "",
            "manipulation_assessment": f"Error analyzing media: {str(e)}",
            "confidence": 0,
            "severity": "UNKNOWN",
            "status": "INCONCLUSIVE",
            "evidence": [],
            "limitations": "Failed to process media"
        }
    
    try:
        genai.delete_file(uploaded_file.name)
    except:
        pass
        
    return data

@app.post("/investigate/media")
async def investigate_media(file: UploadFile = File(...), mode: str = Form("image"), language: str = Form("auto")):
    tmp_path = None
    try:
        suffix = f".{file.filename.split('.')[-1]}"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        mime_map = {
            ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp",
            ".mp4": "video/mp4", ".mov": "video/quicktime", ".webm": "video/webm"
        }
        mime_type = mime_map.get(suffix.lower(), "application/octet-stream")
        
        media_forensics = analyze_media_with_gemini(tmp_path, mime_type, mode)
        transcript = media_forensics.get("ocr_text", "")
        
        if len(transcript.strip()) < 5:
            transcript = f"Media upload ({mode}). Manipulation Status: {media_forensics.get('status')}. Assessment: {media_forensics.get('manipulation_assessment')}"
        
        result = await run_investigation_pipeline(transcript, file.filename, language)
        result["media_forensics"] = media_forensics
        
        return result
    except Exception as e:
        return {"error": str(e)}
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

@app.post("/ask")
async def ask_sona(req: AskRequest):
    try:
        # Ground the LLM by summarizing the current investigation
        report_summary = json.dumps(req.report, indent=2)
        
        response = run_cf_ai(
            max_tokens=500,
            messages=[{
                "role": "system",
                "content": "You are SONA, Pakistan's AI investigation assistant. You answer questions strictly based on the provided investigation report context. Do NOT invent facts. If a user asks about safety, scams, fraud, or what to do next, you MUST strongly advise them to contact the FIA Cybercrime Helpline at 9911 or the State Bank of Pakistan at 0800-55055. Always respond in Roman Urdu if the user's question is in Roman Urdu or Urdu. Be helpful and empathetic."
            }, {
                "role": "user",
                "content": f"Here is the investigation report context:\n{report_summary}\n\nQuestion: {req.query}"
            }]
        )
        return {"answer": response}
    except Exception as e:
        return {"error": str(e)}

# --- WHATSAPP CLOUD API INTEGRATION ---
from fastapi import Request, HTTPException, BackgroundTasks

WHATSAPP_ACCESS_TOKEN = os.getenv("WHATSAPP_ACCESS_TOKEN")
WHATSAPP_PHONE_NUMBER_ID = os.getenv("WHATSAPP_PHONE_NUMBER_ID")
WHATSAPP_VERIFY_TOKEN = os.getenv("WHATSAPP_VERIFY_TOKEN", "sona_verify_token_123")

@app.get("/webhook/whatsapp")
async def verify_whatsapp_webhook(request: Request):
    """Handles Meta webhook verification."""
    mode = request.query_params.get("hub.mode")
    token = request.query_params.get("hub.verify_token")
    challenge = request.query_params.get("hub.challenge")

    if mode and token:
        if mode == "subscribe" and token == WHATSAPP_VERIFY_TOKEN:
            print("WhatsApp WEBHOOK_VERIFIED")
            return int(challenge)
        else:
            raise HTTPException(status_code=403, detail="Verification failed")
    raise HTTPException(status_code=400, detail="Invalid request")

def send_whatsapp_message(to_phone: str, message: str):
    if not WHATSAPP_ACCESS_TOKEN or not WHATSAPP_PHONE_NUMBER_ID:
        print(f"[Simulated WhatsApp Send to {to_phone}]: {message}")
        return
        
    url = f"https://graph.facebook.com/v17.0/{WHATSAPP_PHONE_NUMBER_ID}/messages"
    headers = {
        "Authorization": f"Bearer {WHATSAPP_ACCESS_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {
        "messaging_product": "whatsapp",
        "to": to_phone,
        "type": "text",
        "text": {"body": message}
    }
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code != 200:
        print(f"Failed to send WhatsApp message: {response.text}")

def process_whatsapp_message(message_data: dict):
    # Determine intent and generate a response
    phone_number = message_data.get("from")
    text_body = message_data.get("text", {}).get("body", "").upper()
    
    # Mocking standard commands to handle text inputs
    if text_body == "HELP":
        reply = "SONA WhatsApp Assistant\nCommands:\n- HELP: List commands\n- SUMMARY: Latest summary\n- INVESTIGATE [claim]: Check a claim\n\nOr send a Voice Note to investigate audio."
    elif text_body.startswith("INVESTIGATE"):
        reply = "Investigating your claim through SONA intelligence pipeline... (This is a stub response. Check backend for full integration)."
    else:
        reply = "Received your message. Send a Voice Note or type 'HELP' for commands."
        
    send_whatsapp_message(phone_number, reply)

@app.post("/webhook/whatsapp")
async def handle_whatsapp_webhook(request: Request, background_tasks: BackgroundTasks):
    """Receives incoming WhatsApp messages/status updates."""
    try:
        body = await request.json()
        
        # Log received payload without exposing secrets
        print("WhatsApp Webhook Event Received")
        
        if body.get("object"):
            if "entry" in body and body["entry"][0].get("changes"):
                change = body["entry"][0]["changes"][0]
                value = change.get("value", {})
                
                # Check if it's a message
                if "messages" in value and value["messages"]:
                    message = value["messages"][0]
                    # Process asynchronously
                    background_tasks.add_task(process_whatsapp_message, message)
                    
            return {"status": "success"}
        else:
            raise HTTPException(status_code=404)
    except Exception as e:
        print(f"WhatsApp Webhook Error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
