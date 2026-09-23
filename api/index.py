import os
import sys
import json
import traceback
import urllib.request
import urllib.error
from io import StringIO
from typing import List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Code Interpreter API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class CodeRequest(BaseModel):
    code: str

class CodeResponse(BaseModel):
    error: List[int]
    result: str


def execute_python_code(code: str) -> dict:
    """Executes Python code and captures exact stdout and traceback."""
    old_stdout = sys.stdout
    sys.stdout = StringIO()

    try:
        exec_globals = {}
        exec(code, exec_globals)
        output = sys.stdout.getvalue()
        return {"success": True, "output": output}
    except Exception:
        output = traceback.format_exc()
        return {"success": False, "output": output}
    finally:
        sys.stdout = old_stdout


def analyze_error_with_ai(code: str, error_traceback: str) -> List[int]:
    """Uses AI Pipe (or fallback extraction) to get exact error lines."""
    api_key = os.environ.get("AIPIPE_API_KEY") or os.environ.get("OPENAI_API_KEY")

    if api_key:
        try:
            url = os.environ.get("AIPIPE_BASE_URL", "https://api.aipipe.org/v1/chat/completions")
            if not url.endswith("/chat/completions"):
                url = url.rstrip("/") + "/chat/completions"

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            }

            payload = {
                "model": os.environ.get("AIPIPE_MODEL", "gpt-4o-mini"),
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a Python debugging assistant. Output strictly a JSON object with key 'error_lines' containing an array of integers representing the 1-indexed line number(s) in the user's code where the error occurred. Example: {\"error_lines\": [3]}"
                    },
                    {
                        "role": "user",
                        "content": f"CODE:\n{code}\n\nTRACEBACK:\n{error_traceback}\n\nExtract the line number(s) in the original user code."
                    }
                ],
                "response_format": {"type": "json_object"}
            }

            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers=headers,
                method="POST"
            )

            with urllib.request.urlopen(req, timeout=10) as response:
                resp_data = json.loads(response.read().decode("utf-8"))
                content = resp_data["choices"][0]["message"]["content"]
                parsed = json.loads(content)
                if "error_lines" in parsed and isinstance(parsed["error_lines"], list):
                    return parsed["error_lines"]
        except Exception as e:
            print("AI call error:", e)

    # Reliable fallback directly from traceback
    extracted_lines = []
    for line in error_traceback.splitlines():
        if "line " in line:
            parts = line.split("line ")
            if len(parts) > 1:
                try:
                    num = int(parts[1].split(",")[0].split()[0])
                    extracted_lines.append(num)
                except ValueError:
                    pass

    return extracted_lines if extracted_lines else [1]


@app.get("/")
def home():
    return {"status": "ok", "message": "Code Interpreter API is running"}


@app.post("/code-interpreter", response_model=CodeResponse)
def code_interpreter(req: CodeRequest):
    exec_result = execute_python_code(req.code)

    if exec_result["success"]:
        return CodeResponse(error=[], result=exec_result["output"])

    error_lines = analyze_error_with_ai(req.code, exec_result["output"])
    return CodeResponse(error=error_lines, result=exec_result["output"])
