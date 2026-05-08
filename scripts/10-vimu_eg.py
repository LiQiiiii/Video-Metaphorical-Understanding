# scripts/stage3_evidence_grounding_eval.py

import os
import json
import base64
import time
from typing import Dict, Any, List

import requests
from openai import OpenAI

from scripts.utils import load_jsonl, append_jsonl, ensure_dir

PROJECT_ROOT = "/Your/Path/To/ViMU"

EVIDENCE_PATH = os.path.join(PROJECT_ROOT, "metadata", "video_evidence.jsonl")
QUESTION_PATH = os.path.join(PROJECT_ROOT, "metadata", "vimu_eg.jsonl")
OUT_PATH = os.path.join(PROJECT_ROOT, "output", "vimu_eg_predictions.jsonl")

MAX_FRAMES_PER_REQUEST = 8
SLEEP_BETWEEN_CALLS = 1.0

MODEL_SPECS = [
    {
        "name": "gpt-5.4-mini",
        "family": "OpenAI",
        "size_bucket": "small",
        "open_closed": "closed",
        "provider": "openai",
        "api_model": "gpt-5.4-mini",
        "enabled": False,
    },
    {
        "name": "gpt-4.1-nano",
        "family": "OpenAI",
        "size_bucket": "small",
        "open_closed": "closed",
        "provider": "openai",
        "api_model": "gpt-4.1-nano",
        "enabled": False,
    },
    {
        "name": "o4-mini",
        "family": "OpenAI",
        "size_bucket": "small",
        "open_closed": "closed",
        "provider": "openai",
        "api_model": "o4-mini",
        "enabled": False,
    },
    {
        "name": "gpt-5.2",
        "family": "OpenAI",
        "size_bucket": "small",
        "open_closed": "closed",
        "provider": "openai",
        "api_model": "gpt-5.2",
        "enabled": False,
    },
    {
        "name": "qwen3.5-9b",
        "family": "Qwen",
        "size_bucket": "9b",
        "open_closed": "open",
        "provider": "openai_compatible",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
        "api_model": "qwen/qwen3.5-9b",
        "enabled": False,
    },
    {
        "name": "gemini-3-flash-preview",
        "family": "Gemini",
        "size_bucket": "10b+",
        "open_closed": "closed",
        "provider": "openai_compatible",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
        "api_model": "google/gemini-3-flash-preview",
        "enabled": True,
    },
    {
        "name": "grok-4.1-fast",
        "family": "Grok",
        "size_bucket": "10b+",
        "open_closed": "closed",
        "provider": "openai_compatible",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
        "api_model": "x-ai/grok-4.1-fast",
        "enabled": False,
    },
    {
        "name": "mimo-v2-omni",
        "family": "xiaomi",
        "size_bucket": "10b+",
        "open_closed": "closed",
        "provider": "openai_compatible",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
        "api_model": "xiaomi/mimo-v2-omni",
        "enabled": False,
    },
    {
        "name": "seed-2.0-lite",
        "family": "bytedance-seed",
        "size_bucket": "10b+",
        "open_closed": "closed",
        "provider": "openai_compatible",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
        "api_model": "bytedance-seed/seed-2.0-lite",
        "enabled": False,
    },
    {
        "name": "qwen3.5-27b",
        "family": "Qwen",
        "size_bucket": "27b",
        "open_closed": "open",
        "provider": "openai_compatible",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
        "api_model": "qwen/qwen3.5-27b",
        "enabled": True,
    },
    {
        "name": "ministral-14b",
        "family": "ministral",
        "size_bucket": "14b",
        "open_closed": "open",
        "provider": "openai_compatible",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
        "api_model": "mistralai/ministral-14b-2512",
        "enabled": True,
    },
    {
        "name": "ministral-8b",
        "family": "ministral",
        "size_bucket": "8b",
        "open_closed": "open",
        "provider": "openai_compatible",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
        "api_model": "mistralai/ministral-8b-2512",
        "enabled": False,
    },
    {
        "name": "qwen3-vl-32b-instruct",
        "family": "qwen",
        "size_bucket": "32b",
        "open_closed": "open",
        "provider": "openai_compatible",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
        "api_model": "qwen/qwen3-vl-32b-instruct",
        "enabled": False,
    },
    {
        "name": "glm-4.5v",
        "family": "glm",
        "size_bucket": "10b+",
        "open_closed": "open",
        "provider": "openai_compatible",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
        "api_model": "z-ai/glm-4.5v",
        "enabled": False,
    },
    {
        "name": "gemma-3-4b-it",
        "family": "gemma",
        "size_bucket": "4b",
        "open_closed": "open",
        "provider": "openai_compatible",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
        "api_model": "google/gemma-3-4b-it",
        "enabled": False,
    },
    {
        "name": "gemma-3-27b-it",
        "family": "gemma",
        "size_bucket": "27b",
        "open_closed": "open",
        "provider": "openai_compatible",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
        "api_model": "google/gemma-3-27b-it",
        "enabled": False,
    },
    {
        "name": "claude-3-haiku",
        "family": "anthropic",
        "size_bucket": "10b+",
        "open_closed": "open",
        "provider": "openai_compatible",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
        "api_model": "anthropic/claude-3-haiku",
        "enabled": False,
    },
]

MCQ_SCHEMA = {
    "type": "object",
    "properties": {
        "selected_options": {
            "type": "array",
            "items": {"type": "string"}
        }
    },
    "required": ["selected_options"],
    "additionalProperties": False
}


def image_to_data_url(image_path: str) -> str:
    ext = os.path.splitext(image_path)[1].lower().replace(".", "")
    mime = "jpeg" if ext in ["jpg", "jpeg"] else ext
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/{mime};base64,{b64}"


def format_options(options_dict: Dict[str, Any]) -> str:
    """
    Support either:
    1) {"A": "frames", "B": "visible_text"}
    2) {"A": {"label": "frames", "text": "visual scene / objects"}, ...}
    """
    lines = []
    for key, value in options_dict.items():
        if isinstance(value, dict):
            label = value.get("label", "")
            text = value.get("text", label)
            lines.append(f"{key}. {text} ({label})")
        else:
            lines.append(f"{key}. {value}")
    return "\n".join(lines)


def extract_valid_option_letters(options_dict: Dict[str, Any]) -> List[str]:
    return list(options_dict.keys())


def build_prompt(
    question: str,
    instruction: str,
    options: Dict[str, Any],
    transcript: str,
    intended_meaning: str = "",
    open_ended_question: str = "",
) -> str:
    option_text = format_options(options)

    extra_context = ""
    if open_ended_question:
        extra_context += f"\nRelated interpretation question:\n{open_ended_question}\n"
    if intended_meaning:
        pass

    return f"""You are answering a multi-choice question about a video.

Question:
{question}

Transcript from ASR (may be noisy, partial, or empty):
{transcript}
{extra_context}
Instruction:
{instruction}

Available options:
{option_text}

Return only valid JSON with one field:
{{
  "selected_options": ["A", "B"]
}}

Rules:
- Select only options that are clearly supported by the video.
- Return only option letters, not option texts.
- Do not include any option not in the provided list.
- Be conservative: do not select an option unless it is clearly justified by the video.
- If no option is clearly supported, return an empty list.
"""


def call_openai_model(api_model: str, prompt: str, frame_paths: List[str]) -> Dict[str, Any]:
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    content = [{"type": "input_text", "text": prompt}]
    for fp in frame_paths[:MAX_FRAMES_PER_REQUEST]:
        content.append({
            "type": "input_image",
            "image_url": image_to_data_url(fp)
        })

    resp = client.responses.create(
        model=api_model,
        input=[{
            "role": "user",
            "content": content
        }],
        text={
            "format": {
                "type": "json_schema",
                "name": "evidence_grounding_answer",
                "schema": MCQ_SCHEMA,
                "strict": True
            }
        }
    )
    txt = getattr(resp, "output_text", "") or str(resp)
    return json.loads(txt)


def call_gemini_model(api_model: str, prompt: str, frame_paths: List[str]) -> Dict[str, Any]:
    api_key = os.environ["GOOGLE_API_KEY"]
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{api_model}:generateContent?key={api_key}"

    parts = [{"text": prompt}]
    for fp in frame_paths[:MAX_FRAMES_PER_REQUEST]:
        with open(fp, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode("utf-8")
        parts.append({
            "inline_data": {
                "mime_type": "image/jpeg",
                "data": img_b64
            }
        })

    payload = {
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {
            "responseMimeType": "application/json",
        }
    }

    r = requests.post(url, json=payload, timeout=180)
    r.raise_for_status()
    data = r.json()

    txt = data["candidates"][0]["content"]["parts"][0]["text"]
    return json.loads(txt)


def extract_json_from_chat_completion_content(content: Any) -> Dict[str, Any]:
    if isinstance(content, str):
        return json.loads(content)

    if isinstance(content, list):
        texts = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text" and block.get("text"):
                    texts.append(block["text"])
            else:
                text_attr = getattr(block, "text", None)
                if text_attr:
                    texts.append(text_attr)
        if texts:
            return json.loads("\n".join(texts).strip())

    raise ValueError(f"Could not parse JSON content from response: {content}")


def call_openai_compatible_model(
    base_url: str,
    api_key_env: str,
    api_model: str,
    prompt: str,
    frame_paths: List[str],
) -> Dict[str, Any]:
    api_key = os.environ[api_key_env]

    default_headers = {}
    if "openrouter.ai" in base_url:
        default_headers = {
            "HTTP-Referer": "https://vimu-benchmark.local",
            "X-Title": "ViMUEvidenceGrounding"
        }

    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
        default_headers=default_headers
    )

    content = [{"type": "text", "text": prompt}]
    for fp in frame_paths[:MAX_FRAMES_PER_REQUEST]:
        content.append({
            "type": "image_url",
            "image_url": {"url": image_to_data_url(fp)}
        })

    resp = client.chat.completions.create(
        model=api_model,
        messages=[{
            "role": "user",
            "content": content
        }],
        max_tokens=512,
        response_format={"type": "json_object"},
    )

    if not getattr(resp, "choices", None):
        raise ValueError(f"No choices returned: {resp}")

    msg = resp.choices[0].message
    content_field = getattr(msg, "content", None)

    return extract_json_from_chat_completion_content(content_field)


def run_model(spec: Dict[str, Any], prompt: str, frame_paths: List[str]) -> Dict[str, Any]:
    provider = spec["provider"]

    if provider == "openai":
        return call_openai_model(spec["api_model"], prompt, frame_paths)

    if provider == "gemini":
        return call_gemini_model(spec["api_model"], prompt, frame_paths)

    if provider == "openai_compatible":
        return call_openai_compatible_model(
            base_url=spec["base_url"],
            api_key_env=spec["api_key_env"],
            api_model=spec["api_model"],
            prompt=prompt,
            frame_paths=frame_paths,
        )

    raise ValueError(f"Unknown provider: {provider}")


def clean_prediction(pred: Dict[str, Any], options: Dict[str, Any]) -> List[str]:
    valid_letters = set(options.keys())
    vals = pred.get("selected_options", [])
    if not isinstance(vals, list):
        vals = []

    vals = [str(x).strip() for x in vals if str(x).strip() in valid_letters]

    out = []
    seen = set()
    for x in vals:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def main():
    ensure_dir(os.path.dirname(OUT_PATH))

    evidence_rows = load_jsonl(EVIDENCE_PATH)
    question_rows = load_jsonl(QUESTION_PATH)

    evidence_map = {r["video_id"]: r for r in evidence_rows}
    existing = {
        (r["video_id"], r["model_name"])
        for r in load_jsonl(OUT_PATH)
    }

    active_models = [m for m in MODEL_SPECS if m["enabled"]]

    print(f"Loaded {len(evidence_rows)} evidence rows.")
    print(f"Loaded {len(question_rows)} evidence-grounding questions.")
    print(f"Active models: {[m['name'] for m in active_models]}")

    for row in question_rows:
        video_id = row["video_id"]
        if video_id not in evidence_map:
            print(f"[skip] no evidence for {video_id}")
            continue

        transcript = evidence_map[video_id].get("transcript", "")
        frame_paths = evidence_map[video_id].get("frames", [])

        question = row["question"]
        options = row["options"]
        instruction = row.get(
            "instruction",
            "Select all correct options. Return only the option letters exactly as given. Do not add explanations."
        )
        intended_meaning = row.get("intended_meaning", "")
        open_ended_question = row.get("open_ended_question", "")

        prompt = build_prompt(
            question=question,
            instruction=instruction,
            options=options,
            transcript=transcript,
            intended_meaning=intended_meaning,
            open_ended_question=open_ended_question,
        )

        for spec in active_models:
            key = (video_id, spec["name"])
            if key in existing:
                continue

            try:
                raw_pred = run_model(spec, prompt, frame_paths)
                cleaned = clean_prediction(raw_pred, options)

                append_jsonl(OUT_PATH, {
                    "video_id": video_id,
                    "model_name": spec["name"],
                    "task_name": "evidence_grounding",
                    "prediction": {
                        "selected_options": cleaned
                    }
                })
                print(f"[ok] {video_id} | {spec['name']}")
            except Exception as e:
                append_jsonl(OUT_PATH, {
                    "video_id": video_id,
                    "model_name": spec["name"],
                    "task_name": "evidence_grounding",
                    "prediction": {
                        "selected_options": [],
                        "error": str(e)
                    }
                })
                print(f"[error] {video_id} | {spec['name']} | {e}")

            time.sleep(SLEEP_BETWEEN_CALLS)


if __name__ == "__main__":
    main()