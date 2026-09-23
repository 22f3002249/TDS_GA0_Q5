import os
import sys
import io
import traceback
import json
from typing import List
from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI

app = FastAPI()

# Enable CORS for the evaluator
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# AI Pipe Setup
# The token will be pulled from Vercel Environment Variables
client = OpenAI(
    base_url="https://aipipe.org/openrouter/v1",
    api_key=os.environ.get("AIPIPE_TOKEN")
)

class CodeRequest(BaseModel):
    code: str

def execute_python_code(code: str) -> dict:
    old_stdout = sys.stdout
    redirected_output = sys.stdout = io.StringIO()
    try:
        # Using a dictionary for globals to persist state across lines
        exec_scope = {}
        exec(code, exec_scope)
        output = redirected_output.getvalue()
        return {"success": True, "output": output}
    except Exception:
        # Capture the standard traceback string
        output = traceback.format_exc()
        return {"success": False, "output": output}
    finally:
        sys.stdout = old_stdout

def analyze_error_with_ai(code: str, error_traceback: str) -> List[int]:
    prompt = f"""
Analyze this Python code and its error traceback.
Identify the line number(s) (1-indexed) where the error occurred.

CODE:
{code}

TRACEBACK:
{error_traceback}

Return ONLY a JSON object with the key "error_lines" containing a list of integers.
Example: {{"error_lines": [3]}}
"""
    try:
        response = client.chat.completions.create(
            model="google/gemini-2.0-flash-lite-001",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        content = response.choices[0].message.content
        result = json.loads(content)
        return result.get("error_lines", [])
    except:
        return []

@app.post("/code-interpreter")
async def interpreter(payload: CodeRequest):
    execution = execute_python_code(payload.code)
    
    if execution["success"]:
        return {"error": [], "result": execution["output"]}
    
    # Only call AI if execution failed
    error_lines = analyze_error_with_ai(payload.code, execution["output"])
    return {"error": error_lines, "result": execution["output"]}

@app.get("/")
async def health():
    return {"status": "interpreter online"}
