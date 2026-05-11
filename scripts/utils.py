import os
import json
import time
import base64
from pathlib import Path
from typing import Any, Dict, List, Optional
 
from openai import OpenAI

api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not set")

client = OpenAI(api_key=api_key)


def ensure_dir(path: str):
    if path:
        Path(path).mkdir(parents=True, exist_ok=True)


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: str, row: Dict[str, Any]):
    ensure_dir(os.path.dirname(path))
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def image_to_data_url(image_path: str) -> str:
    ext = Path(image_path).suffix.lower().replace(".", "")
    mime = "jpeg" if ext in ["jpg", "jpeg"] else ext
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/{mime};base64,{b64}"


def response_text(resp) -> str:
    if hasattr(resp, "output_text") and resp.output_text:
        return resp.output_text
    try:
        return resp.output[0].content[0].text
    except Exception:
        return str(resp)


def call_structured_response(
    model: str,
    system_prompt: str,
    user_text: str,
    image_paths: Optional[List[str]],
    schema_name: str,
    schema: Dict[str, Any],
    max_retries: int = 3,
    sleep_seconds: float = 1.0,
):
    content = [{"type": "input_text", "text": user_text}]
    if image_paths:
        for p in image_paths:
            content.append({
                "type": "input_image",
                "image_url": image_to_data_url(p)
            })

    last_err = None
    for attempt in range(max_retries):
        try:
            resp = client.responses.create(
                model=model,
                input=[
                    {
                        "role": "system",
                        "content": [{"type": "input_text", "text": system_prompt}]
                    },
                    {
                        "role": "user",
                        "content": content
                    }
                ],
                text={
                    "format": {
                        "type": "json_schema",
                        "name": schema_name,
                        "schema": schema,
                        "strict": True
                    }
                }
            )
            txt = response_text(resp)
            return json.loads(txt)
        except Exception as e:
            last_err = e
            time.sleep(sleep_seconds * (attempt + 1))

    raise last_err
