import os
import json
import base64
import time
from typing import Dict, Any, List
import argparse

import requests
from openai import OpenAI

from scripts.utils import load_jsonl, append_jsonl, ensure_dir

PROJECT_ROOT = "/Your/Path/To/ViMU"

EVIDENCE_PATH = os.path.join(PROJECT_ROOT, "metadata", "video_evidence.jsonl")
MCQ_PATH = os.path.join(PROJECT_ROOT, "metadata", "vimu_ss.jsonl")
out_path = os.path.join(PROJECT_ROOT, "metadata", "vimu_ss_predictions.jsonl")

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


def format_options(options: Dict[str, str]) -> str:
    lines = []
    for key, value in options.items():
        lines.append(f"{key}. {value}")
    return "\n".join(lines)


def build_prompt_without_guidance(
    task_name: str,
    question: str,
    instruction: str,
    options: Dict[str, str],
    transcript: str
) -> str:
    option_text = format_options(options)

    return f"""You are answering a multi-choice question about a video.

Task name:
{task_name}

Question:
{question}

Transcript from ASR (may be noisy, partial, or empty):
{transcript}

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


def build_prompt_with_guidance(
    task_name: str,
    question: str,
    instruction: str,
    options: Dict[str, str],
    transcript: str
) -> str:
    option_text = format_options(options)

    RHETORIC_GUIDANCE = """
Additional guidance on taxonomy categories:

The options correspond to high-level rhetorical categories.
Each category summarizes several finer-grained rhetorical patterns commonly observed in video memes.

Rhetoric Macro Categories:

A. Literal / Direct
Meaning is conveyed directly without rhetorical transformation.
Typical patterns include:
- literal_only: the video communicates its message directly without irony, exaggeration, or figurative framing.

B. Opposition / Incongruity
Meaning arises from contradiction, reversal, or unexpected juxtaposition.
Typical patterns include:
- contrast: juxtaposing two opposing situations, ideas, or outcomes.
- bait_and_switch: setting up one expectation and then suddenly replacing it with a different or contradictory outcome.
- role_reversal: reversing expected roles, identities, or positions to produce humor or commentary.
- absurdism: presenting illogical or exaggerated situations that highlight incongruity.

C. Attitude / Tone-based Rhetoric
Meaning is conveyed primarily through tone or speaker attitude.
Typical patterns include:
- sarcasm: expressing a meaning by stating the opposite of what is intended.
- irony: the intended meaning contrasts with the literal situation or appearance.
- deadpan: presenting absurd or humorous content in a serious, emotionless manner.
- mockery: ridiculing or making fun of a person, behavior, or situation.

D. Amplification / Stylization
Meaning is emphasized through exaggeration or stylized imitation.
Typical patterns include:
- exaggeration: overstating a situation or characteristic to emphasize its significance.
- parody: imitating the style or conventions of a person, genre, or cultural artifact for humorous or critical effect.

E. Implicit / Coded Social Framing
Meaning is conveyed through indirect or socially coded signals.
Typical patterns include:
- innuendo: suggesting a meaning indirectly rather than stating it explicitly.
- stereotype_invocation: referencing widely known stereotypes to imply a social meaning.
- dog_whistle_or_code: using coded expressions that convey specific meanings to certain audiences while remaining subtle to others.

When answering:
- Focus on how the video constructs its underlying meaning.
- Identify the rhetorical strategies used to convey humor, critique, or commentary.
- A video may contain multiple rhetorical mechanisms.
"""

    SOCIAL_GUIDANCE = """
Additional guidance on taxonomy categories:

The options correspond to high-level social value signals.
Each category summarizes several finer-grained social attitudes or stances that may be expressed in video memes.

Social Value Signal Categories:

A. Neutral / No Social Signal
The video conveys humor or content without expressing a clear social stance.
Typical patterns include:
- none: the video does not convey a noticeable social judgment, stance, or value signal.

B. Emotional Attitude
The video expresses a general emotional tone or affect toward a situation.
Typical patterns include:
- negative_affect: expressing frustration, disappointment, annoyance, or dissatisfaction.
- fatalism_or_cynicism: expressing pessimism, resignation, or cynical attitudes about situations or outcomes.

C. Social Evaluation / Devaluation
The video evaluates, criticizes, or demeans people or groups.
Typical patterns include:
- contempt: expressing disdain or disrespect toward someone or something.
- humiliation: portraying someone as foolish, incompetent, or inferior.
- aggression_or_hostility: showing hostility, threats, or aggressive attitudes.
- exclusion: implying that certain people or groups should be excluded or marginalized.
- discrimination_or_prejudice: expressing biased or discriminatory attitudes toward social groups.

D. Norm and Value Framing
The video comments on social rules, expectations, or cultural norms.
Typical patterns include:
- norm_violation: highlighting or mocking behavior that breaks accepted social rules.
- anti_mainstream_value: expressing opposition to widely accepted social norms or values.

E. Identity / Ideological Signaling
The video references identity, ideology, or social group affiliation.
Typical patterns include:
- political_or_identity_signal: expressing political stances or identity-based perspectives.
- sexual_implication: implying sexual themes or identity-related meanings.
- other: conveying social signals that do not clearly fall into the above categories.

When answering:
- Focus on what social stance, value judgment, or attitude the video conveys.
- Identify whether the video expresses opinions about people, groups, norms, or identities.
- A video may express multiple social value signals.
"""

    if "rhetoric" in task_name:
        taxonomy_guidance = RHETORIC_GUIDANCE
    else:
        taxonomy_guidance = SOCIAL_GUIDANCE

    return f"""You are analyzing a video and answering a multi-choice question.

Your goal is to identify the most appropriate categories based on the video's meaning.

Task name:
{task_name}

Question:
{question}

Transcript from ASR (may be noisy, partial, or empty):
{transcript}

{taxonomy_guidance}

Instruction:
{instruction}

Available options:
{option_text}

Output format:
Return ONLY valid JSON with the following structure:
{{
  "selected_options": ["A", "B"]
}}

Rules:
- Only return option letters (A–E).
- Do NOT return option texts.
- Select all options that are clearly supported by the video.
- Do NOT guess if evidence is weak.
- If none apply, return an empty list.
"""


def build_prompt(
    task_name: str,
    question: str,
    instruction: str,
    options: Dict[str, str],
    transcript: str,
    prompt_mode: str,
) -> str:
    if prompt_mode == "with_guidance":
        return build_prompt_with_guidance(
            task_name, question, instruction, options, transcript
        )

    if prompt_mode == "without_guidance":
        return build_prompt_without_guidance(
            task_name, question, instruction, options, transcript
        )

    raise ValueError(f"Unknown prompt_mode: {prompt_mode}")

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
                "name": "taxonomy_macro_mcq_answer",
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
            "X-Title": "ViMUSSMCQ"
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


def clean_prediction(pred: Dict[str, Any], options: Dict[str, str]) -> List[str]:
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

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt_mode",
        type=str,
        default="with_guidance",
        choices=["with_guidance", "without_guidance"],
        help="Whether to use taxonomy guidance in the prompt."
    )

    parser.add_argument(
        "--out_path",
        type=str,
        default=None,
        help="Optional output path. If not set, chosen based on prompt_mode."
    )

    return parser.parse_args()

def main():
    args = parse_args()

    if args.out_path is not None:
        out_path = args.out_path
    else:
        suffix = "with_guidance" if args.prompt_mode == "with_guidance" else "without_guidance"
        out_path = os.path.join(
            PROJECT_ROOT,
            "output",
            f"vimu_ss_predictions_{suffix}.jsonl"
        )

    ensure_dir(os.path.dirname(out_path))

    evidence_rows = load_jsonl(EVIDENCE_PATH)
    mcq_rows = load_jsonl(MCQ_PATH)

    evidence_map = {r["video_id"]: r for r in evidence_rows}
    existing = {
        (r["video_id"], r["model_name"], r["task_name"])
        for r in load_jsonl(out_path)
    }

    active_models = [m for m in MODEL_SPECS if m["enabled"]]

    print(f"Loaded {len(evidence_rows)} evidence rows.")
    print(f"Loaded {len(mcq_rows)} MCQ rows.")
    print(f"Active models: {[m['name'] for m in active_models]}")

    for row in mcq_rows:
        video_id = row["video_id"]
        if video_id not in evidence_map:
            print(f"[skip] no evidence for {video_id}")
            continue

        transcript = evidence_map[video_id].get("transcript", "")
        frame_paths = evidence_map[video_id].get("frames", [])
        mcq_tasks = row["mcq_tasks"]

        for task_name, task in mcq_tasks.items():
            prompt = build_prompt(
                task_name=task_name,
                question=task["question"],
                instruction=task["instruction"],
                options=task["options"],
                transcript=transcript,
                prompt_mode=args.prompt_mode,
            )

            for spec in active_models:
                key = (video_id, spec["name"], task_name)
                if key in existing:
                    continue

                try:
                    raw_pred = run_model(spec, prompt, frame_paths)
                    cleaned = clean_prediction(raw_pred, task["options"])

                    append_jsonl(out_path, {
                        "video_id": video_id,
                        "model_name": spec["name"],
                        "task_name": task_name,
                        "prediction": {
                            "selected_options": cleaned
                        }
                    })
                    print(f"[ok] {video_id} | {spec['name']} | {task_name}")
                except Exception as e:
                    append_jsonl(out_path, {
                        "video_id": video_id,
                        "model_name": spec["name"],
                        "task_name": task_name,
                        "prediction": {
                            "selected_options": [],
                            "error": str(e)
                        }
                    })
                    print(f"[error] {video_id} | {spec['name']} | {task_name} | {e}")

                time.sleep(SLEEP_BETWEEN_CALLS)


if __name__ == "__main__":
    main()