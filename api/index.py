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
        # Execute code in clean global scope
        exec(code, {})
        output = sys.stdout.getvalue()
        return {"success": True, "output": output}
    except Exception:
        output = traceback.format_exc()
        return {"success": False, "output": output}
    finally:
        sys.stdout = old_stdout


def extract_lines_from_traceback(code: str, error_traceback: str) -> List[int]:
    """Extract line numbers specifically from the user code execution frame (<string>)."""
    total_lines = len(code.splitlines())
    matched_lines = []

    for line in error_traceback.splitlines():
        # Only parse frames belonging to the executed string, not server files
        if '<string>' in line and 'line ' in line:
            parts = line.split("line ")
            if len(parts) > 1:
                try:
                    num_str = parts[1].split(",")[0].split()[0]
                    num = int(num_str)
                    if 1 <= num <= total_lines and num not in matched_lines:
                        matched_lines.append(num)
                except ValueError:
                    pass

    return matched_lines


def analyze_error_with_ai(code: str, error_traceback: str) -> List[int]:
    """Uses AI to identify exact error line numbers within the user's code."""
    total_lines = len(code.splitlines())
    numbered_code = "\n".join([f"{i+1}: {line}" for i, line in enumerate(code.splitlines())])

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

            prompt = f"""You are a Python error analysis system.
Given the numbered user code and the error traceback, find the line number(s) in the USER CODE where the error originated.

NUMBERED USER CODE:
{numbered_code}

TRACEBACK:
{error_traceback}

IMPORTANT RULES:
- The user code only has {total_lines} line(s).
- Return ONLY valid line numbers between 1 and {total_lines}.
- Do NOT include line numbers from server or library files.
- Return JSON strictly matching: {{"error_lines": [line_number]}}
"""

            payload = {
                "model": os.environ.get("AIPIPE_MODEL", "gpt-4o-mini"),
                "messages": [
                    {"role": "system", "content": "You return valid JSON with error line numbers for Python code."},
                    {"role": "user", "content": prompt}
                ],
                "response_format": {"type": "json_object"}
            }

            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers=headers,
                method="POST"
            )

            with urllib.request.urlopen(req, timeout=8) as response:
                resp_data = json.loads(response.read().decode("utf-8"))
                content = resp_data["choices"][0]["message"]["content"]
                parsed = json.loads(content)
                if "error_lines" in parsed and isinstance(parsed["error_lines"], list):
                    # Filter strictly within user code line range
                    filtered = [n for n in parsed["error_lines"] if isinstance(n, int) and 1 <= n <= total_lines]
                    if filtered:
                        return filtered
        except Exception as e:
            print("AI Pipe error:", e)

    # Accurate traceback-based fallback
    tb_lines = extract_lines_from_traceback(code, error_traceback)
    if tb_lines:
        return tb_lines

    return [1] if total_lines >= 1 else []


@app.get("/")
def health():
    return {"status": "ok"}


@app.post("/code-interpreter", response_model=CodeResponse)
def code_interpreter(req: CodeRequest):
    exec_result = execute_python_code(req.code)

    if exec_result["success"]:
        return CodeResponse(error=[], result=exec_result["output"])

    error_lines = analyze_error_with_ai(req.code, exec_result["output"])
    return CodeResponse(error=error_lines, result=exec_result["output"])
