import os
import json
import time
import base64
from typing import Dict, Any, List, Optional

import requests

from openai import OpenAI
from scripts.utils import load_jsonl, append_jsonl, ensure_dir


PROJECT_ROOT = "/Your/Path/To/ViMU"

EVIDENCE_PATH = os.path.join(PROJECT_ROOT, "metadata", "video_evidence.jsonl")
QA_PATH = os.path.join(PROJECT_ROOT, "metadata", "vimu_oe.jsonl")
OUT_PATH = os.path.join(PROJECT_ROOT, "output", "vimu_oe_predictions.jsonl")


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
        "enabled": True,
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

MAX_FRAMES_PER_REQUEST = 8
SLEEP_BETWEEN_CALLS = 1.0

MAX_RETRIES_PER_SAMPLE = 8
RETRY_SLEEP_SECONDS = 2.0
MIN_FRAMES_PER_REQUEST = 1


def image_to_data_url(image_path: str) -> str:
    ext = os.path.splitext(image_path)[1].lower().replace(".", "")
    mime = "jpeg" if ext in ["jpg", "jpeg"] else ext

    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")

    return f"data:image/{mime};base64,{b64}"


def build_video_prompt(question: str, transcript: str) -> str:
    return f"""You are answering a benchmark question about a video.

Question:
{question}

Transcript from ASR (may be noisy, partial, or empty):
{transcript}

Instructions:
- Answer the question directly.
- Use the video frames as the primary source of truth.
- Infer visible on-screen text from the frames when relevant.
- Keep the answer concise but semantically complete.
- Do not add safety disclaimers unless absolutely necessary.
"""


def call_openai_model(
    api_model: str,
    question: str,
    transcript: str,
    frame_paths: List[str],
) -> str:
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    content = [{"type": "input_text", "text": build_video_prompt(question, transcript)}]

    for fp in frame_paths[:MAX_FRAMES_PER_REQUEST]:
        content.append({
            "type": "input_image",
            "image_url": image_to_data_url(fp),
        })

    resp = client.responses.create(
        model=api_model,
        input=[{
            "role": "user",
            "content": content,
        }],
    )

    return getattr(resp, "output_text", "") or str(resp)


def call_gemini_model(
    api_model: str,
    question: str,
    transcript: str,
    frame_paths: List[str],
) -> str:
    api_key = os.environ["GOOGLE_API_KEY"]
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{api_model}:generateContent?key={api_key}"

    parts = [{"text": build_video_prompt(question, transcript)}]

    for fp in frame_paths[:MAX_FRAMES_PER_REQUEST]:
        with open(fp, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode("utf-8")

        parts.append({
            "inline_data": {
                "mime_type": "image/jpeg",
                "data": img_b64,
            }
        })

    payload = {
        "contents": [
            {
                "role": "user",
                "parts": parts,
            }
        ]
    }

    r = requests.post(url, json=payload, timeout=180)
    r.raise_for_status()
    data = r.json()

    try:
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except Exception:
        return json.dumps(data, ensure_ascii=False)


def call_anthropic_model(
    api_model: str,
    question: str,
    transcript: str,
    frame_paths: List[str],
) -> str:
    api_key = os.environ["ANTHROPIC_API_KEY"]
    url = "https://api.anthropic.com/v1/messages"

    content = [{
        "type": "text",
        "text": build_video_prompt(question, transcript),
    }]

    for fp in frame_paths[:MAX_FRAMES_PER_REQUEST]:
        with open(fp, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode("utf-8")

        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": img_b64,
            },
        })

    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }

    payload = {
        "model": api_model,
        "max_tokens": 512,
        "messages": [
            {
                "role": "user",
                "content": content,
            }
        ],
    }

    r = requests.post(url, headers=headers, json=payload, timeout=180)
    r.raise_for_status()
    data = r.json()

    try:
        texts = [x["text"] for x in data["content"] if x["type"] == "text"]
        return "\n".join(texts).strip()
    except Exception:
        return json.dumps(data, ensure_ascii=False)


def call_openai_compatible_model(
    base_url: str,
    api_key_env: str,
    api_model: str,
    question: str,
    transcript: str,
    frame_paths: List[str],
) -> str:
    api_key = os.environ[api_key_env]
    client = OpenAI(api_key=api_key, base_url=base_url)

    content = [{"type": "text", "text": build_video_prompt(question, transcript)}]

    for fp in frame_paths[:MAX_FRAMES_PER_REQUEST]:
        content.append({
            "type": "image_url",
            "image_url": {"url": image_to_data_url(fp)},
        })

    resp = client.chat.completions.create(
        model=api_model,
        messages=[{
            "role": "user",
            "content": content,
        }],
        max_tokens=2048,
    )

    if not getattr(resp, "choices", None):
        try:
            return json.dumps(resp.model_dump(), ensure_ascii=False)
        except Exception:
            return str(resp)

    choice = resp.choices[0]
    msg = getattr(choice, "message", None)

    if msg is None:
        try:
            return json.dumps(resp.model_dump(), ensure_ascii=False)
        except Exception:
            return str(resp)

    content_field = getattr(msg, "content", None)

    if isinstance(content_field, str) and content_field.strip():
        return content_field.strip()

    if isinstance(content_field, list):
        texts = []
        for block in content_field:
            if isinstance(block, dict):
                if block.get("type") == "text" and block.get("text"):
                    texts.append(block["text"])
            else:
                text_attr = getattr(block, "text", None)
                if text_attr:
                    texts.append(text_attr)

        if texts:
            return "\n".join(texts).strip()

    reasoning_field = getattr(msg, "reasoning", None)

    if reasoning_field:
        if isinstance(reasoning_field, str):
            return f"[ONLY_REASONING_RETURNED]\n{reasoning_field}"

        try:
            return f"[ONLY_REASONING_RETURNED]\n{json.dumps(reasoning_field, ensure_ascii=False)}"
        except Exception:
            return f"[ONLY_REASONING_RETURNED]\n{str(reasoning_field)}"

    try:
        return json.dumps(resp.model_dump(), ensure_ascii=False)
    except Exception:
        return str(resp)


def run_model(
    spec: Dict[str, Any],
    question: str,
    transcript: str,
    frame_paths: List[str],
    max_frames: Optional[int] = None,
) -> str:
    provider = spec["provider"]

    if max_frames is not None:
        frame_paths = frame_paths[:max_frames]

    if provider == "openai":
        return call_openai_model(
            spec["api_model"],
            question,
            transcript,
            frame_paths,
        )

    if provider == "gemini":
        return call_gemini_model(
            spec["api_model"],
            question,
            transcript,
            frame_paths,
        )

    if provider == "anthropic":
        return call_anthropic_model(
            spec["api_model"],
            question,
            transcript,
            frame_paths,
        )

    if provider == "openai_compatible":
        return call_openai_compatible_model(
            base_url=spec["base_url"],
            api_key_env=spec["api_key_env"],
            api_model=spec["api_model"],
            question=question,
            transcript=transcript,
            frame_paths=frame_paths,
        )

    raise ValueError(f"Unknown provider: {provider}")


def should_reduce_frames_from_error(err_msg: str) -> Optional[int]:
    if not err_msg:
        return None

    import re

    m = re.search(
        r"Image count\s+\d+\s+exceeds limit\s+(\d+)\s+per request",
        err_msg,
    )

    if m:
        return int(m.group(1))

    return None


def run_model_with_retry(
    spec: Dict[str, Any],
    question: str,
    transcript: str,
    frame_paths: List[str],
) -> str:
    current_max_frames = min(len(frame_paths), MAX_FRAMES_PER_REQUEST)
    last_err = None

    for attempt in range(1, MAX_RETRIES_PER_SAMPLE + 1):
        try:
            return run_model(
                spec=spec,
                question=question,
                transcript=transcript,
                frame_paths=frame_paths,
                max_frames=current_max_frames,
            )

        except Exception as e:
            err_msg = str(e)
            last_err = e

            reduced_limit = should_reduce_frames_from_error(err_msg)

            if reduced_limit is not None:
                new_max_frames = min(current_max_frames, reduced_limit)

                if new_max_frames < current_max_frames:
                    print(
                        f"[retry-adjust] {spec['name']} | "
                        f"frames {current_max_frames} -> {new_max_frames}"
                    )
                    current_max_frames = max(
                        new_max_frames,
                        MIN_FRAMES_PER_REQUEST,
                    )
                    time.sleep(RETRY_SLEEP_SECONDS)
                    continue

            print(
                f"[retry] {spec['name']} | "
                f"attempt {attempt}/{MAX_RETRIES_PER_SAMPLE} | {err_msg}"
            )
            time.sleep(RETRY_SLEEP_SECONDS)

    raise RuntimeError(
        f"Failed after {MAX_RETRIES_PER_SAMPLE} retries. Last error: {last_err}"
    )


def main():
    ensure_dir(os.path.dirname(OUT_PATH))

    evidence_rows = load_jsonl(EVIDENCE_PATH)
    qa_rows = load_jsonl(QA_PATH)

    evidence_map = {
        r["video_id"]: r
        for r in evidence_rows
        if "video_id" in r
    }

    accepted_qas = {
        r["video_id"]: r
        for r in qa_rows
        if "video_id" in r
        and "qa" in r
        and isinstance(r["qa"], dict)
        and "question" in r["qa"]
    }

    existing = {
        (r["video_id"], r["model_name"])
        for r in load_jsonl(OUT_PATH)
        if "video_id" in r and "model_name" in r
    }

    active_models = [m for m in MODEL_SPECS if m.get("enabled")]

    print(f"Loaded {len(evidence_rows)} evidence rows.")
    print(f"Loaded {len(qa_rows)} QA rows from QA_PATH.")
    print(f"Loaded {len(accepted_qas)} usable cleaned QA rows from QA_PATH.")
    print(f"Active models: {[m['name'] for m in active_models]}")

    for video_id, qa_row in accepted_qas.items():
        if video_id not in evidence_map:
            print(f"[skip] no evidence for {video_id}")
            continue

        question = qa_row["qa"]["question"]
        gold_answer = qa_row["qa"].get("answer", "")
        transcript = evidence_map[video_id].get("transcript", "")
        frame_paths = evidence_map[video_id].get("frames", [])

        for spec in active_models:
            key = (video_id, spec["name"])

            if key in existing:
                continue

            try:
                pred = run_model_with_retry(
                    spec=spec,
                    question=question,
                    transcript=transcript,
                    frame_paths=frame_paths,
                )

                append_jsonl(OUT_PATH, {
                    "video_id": video_id,
                    "model_name": spec["name"],
                    "model_family": spec["family"],
                    "size_bucket": spec["size_bucket"],
                    "open_closed": spec["open_closed"],
                    "provider": spec["provider"],
                    "question": question,
                    "gold_answer": gold_answer,
                    "prediction": pred,
                })

                existing.add(key)

                print(f"[ok] {video_id} | {spec['name']}")

            except Exception as e:
                append_jsonl(OUT_PATH, {
                    "video_id": video_id,
                    "model_name": spec["name"],
                    "model_family": spec["family"],
                    "size_bucket": spec["size_bucket"],
                    "open_closed": spec["open_closed"],
                    "provider": spec["provider"],
                    "question": question,
                    "gold_answer": gold_answer,
                    "prediction": f"[ERROR] {e}",
                })

                existing.add(key)

                print(f"[error-final] {video_id} | {spec['name']} | {e}")

            time.sleep(SLEEP_BETWEEN_CALLS)


if __name__ == "__main__":
    main()