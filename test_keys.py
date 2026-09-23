import os
import requests
from dotenv import load_dotenv

load_dotenv()

print("Testing Gemini API Key...")
try:
    gemini_key = os.getenv("GEMINI_API_KEY")
    if not gemini_key:
        print("FAILED: Missing GEMINI_API_KEY")
    else:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={gemini_key}"
        payload = {"contents": [{"parts": [{"text": "Hello"}]}]}
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            print("SUCCESS: Gemini API Key is working via HTTP!")
        else:
            print(f"FAILED: Gemini Error: {response.status_code} - {response.text}")
except Exception as e:
    print(f"FAILED: Gemini Error: {e}")

print("\nTesting Cloudflare Workers AI...")
try:
    account_id = os.getenv("CLOUDFLARE_ACCOUNT_ID")
    api_token = os.getenv("CLOUDFLARE_API_TOKEN")
    if not account_id or not api_token:
        print("FAILED: Missing CLOUDFLARE_ACCOUNT_ID or CLOUDFLARE_API_TOKEN")
    else:
        url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/@cf/meta/llama-3.1-8b-instruct"
        headers = {"Authorization": f"Bearer {api_token}"}
        payload = {"messages": [{"role": "user", "content": "Hello"}], "max_tokens": 10}
        response = requests.post(url, headers=headers, json=payload)
        if response.status_code == 200:
            print("SUCCESS: Cloudflare AI Key is working!")
        else:
            print(f"FAILED: Cloudflare Error: {response.status_code} - {response.text}")
except Exception as e:
    print(f"FAILED: Cloudflare Error: {e}")

print("\nTesting Tavily API Key...")
try:
    search_response = requests.post(
        "https://api.tavily.com/search",
        json={
            "api_key": os.getenv("TAVILY_API_KEY"),
            "query": "Hello",
            "search_depth": "basic",
            "max_results": 1
        }
    )
    if search_response.status_code == 200:
        print("SUCCESS: Tavily API Key is working!")
    else:
        print(f"FAILED: Tavily Error: {search_response.status_code} - {search_response.text}")
except Exception as e:
    print(f"FAILED: Tavily Error: {e}")
