import os
import json

from scripts.utils import load_jsonl, append_jsonl, call_structured_response, ensure_dir


PROJECT_ROOT = "/Your/Path/To/ViMU"

QA_PATH = os.path.join(PROJECT_ROOT, "metadata", "vimu_oe.jsonl")
PRED_PATH = os.path.join(PROJECT_ROOT, "output", "vimu_oe_predictions.jsonl")
OUT_PATH = os.path.join(PROJECT_ROOT, "output", "vimu_oe_judged.jsonl")

JUDGE_MODEL = "gpt-5.4"


JUDGE_SYSTEM_PROMPT = """You are grading answers in a benchmark for hint-free implicit video understanding.

Your job is to judge semantic understanding, not style.

Scoring dimensions:
- core_intent: Did the model capture the main intended meaning?
- implicit_signal: Did it recognize the crucial hidden rhetorical or social signal?
- target_or_social_meaning: Did it identify a relevant target, group, institution, or social implication when supported?
- hallucination_penalty: Penalize invented claims not grounded by the gold answer, evidence, or rubric.
- literal_only_penalty: Penalize answers that remain at surface description and miss the point.

Score bounds:
- core_intent must be one of {0, 1, 2, 3, 4, 5}
- implicit_signal must be one of {0, 1, 2, 3}
- target_or_social_meaning must be one of {0, 1}
- hallucination_penalty must be one of {0, 1, 2, 3}
- literal_only_penalty must be one of {0, 1, 2, 3}

Scoring rule:
score_total =
core_intent
+ implicit_signal
+ target_or_social_meaning
- hallucination_penalty
- literal_only_penalty

The maximum possible score is 9.

Interpretation guide:
- A partially correct answer that captures the meme's point should score much higher than a polished but purely literal answer.
- Do not require exact wording match.
- Be strict with hallucinations.
- Only assign target_or_social_meaning = 1 when that dimension is genuinely relevant and correctly captured.
- Use the evidence as grounding support, not as extra hidden labels to overfit.
- Keep reasoning_short concise.

Return only valid JSON.
"""


JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "score_total": {
            "type": "integer",
            "enum": [-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        },
        "score_breakdown": {
            "type": "object",
            "properties": {
                "core_intent": {
                    "type": "integer",
                    "enum": [0, 1, 2, 3, 4, 5]
                },
                "implicit_signal": {
                    "type": "integer",
                    "enum": [0, 1, 2, 3]
                },
                "target_or_social_meaning": {
                    "type": "integer",
                    "enum": [0, 1]
                },
                "hallucination_penalty": {
                    "type": "integer",
                    "enum": [0, 1, 2, 3]
                },
                "literal_only_penalty": {
                    "type": "integer",
                    "enum": [0, 1, 2, 3]
                }
            },
            "required": [
                "core_intent",
                "implicit_signal",
                "target_or_social_meaning",
                "hallucination_penalty",
                "literal_only_penalty"
            ],
            "additionalProperties": False
        },
        "verdict": {
            "type": "string",
            "enum": ["excellent", "good", "partial", "poor", "wrong"]
        },
        "reasoning_short": {"type": "string"}
    },
    "required": [
        "score_total",
        "score_breakdown",
        "verdict",
        "reasoning_short"
    ],
    "additionalProperties": False
}


def normalize_prediction_text(prediction):
    if isinstance(prediction, str):
        return prediction

    return json.dumps(prediction, ensure_ascii=False)


def build_qa_map(qa_rows):
    qa_map = {}

    for r in qa_rows:
        if "video_id" not in r:
            continue

        qa = r.get("qa", {})
        taxonomy = r.get("taxonomy", {})

        if not isinstance(qa, dict):
            continue

        if "question" not in qa or "answer" not in qa:
            continue

        qa_map[r["video_id"]] = {
            "question": qa.get("question", ""),
            "gold_answer": qa.get("answer", ""),
            "difficulty": qa.get("difficulty", ""),
            "rubric": qa.get("rubric", {}),
            "evidence": taxonomy.get("evidence", []),
            "intended_meaning": taxonomy.get("intended_meaning", ""),
            "literal_summary": taxonomy.get("literal_summary", ""),
        }

    return qa_map


def main():
    ensure_dir(os.path.dirname(OUT_PATH))

    qa_rows = load_jsonl(QA_PATH)
    pred_rows = load_jsonl(PRED_PATH)

    qa_map = build_qa_map(qa_rows)

    existing = {
        (r["video_id"], r["model_name"])
        for r in load_jsonl(OUT_PATH)
        if "video_id" in r and "model_name" in r
    }

    print(f"Loaded {len(qa_map)} cleaned QA rows.")
    print(f"Loaded {len(pred_rows)} prediction rows.")
    print(f"Loaded {len(existing)} existing judged rows.")

    for row in pred_rows:
        video_id = row["video_id"]
        model_name = row["model_name"]

        if (video_id, model_name) in existing:
            continue

        if video_id not in qa_map:
            print(f"[skip] no cleaned QA for {video_id}")
            continue

        qa = qa_map[video_id]
        prediction = normalize_prediction_text(row.get("prediction", ""))

        if isinstance(prediction, str) and prediction.startswith("[ERROR]"):
            append_jsonl(OUT_PATH, {
                "video_id": video_id,
                "model_name": model_name,
                "model_family": row.get("model_family", ""),
                "size_bucket": row.get("size_bucket", ""),
                "open_closed": row.get("open_closed", ""),
                "provider": row.get("provider", ""),
                "question": qa["question"],
                "gold_answer": qa["gold_answer"],
                "prediction": prediction,
                "judgment": {
                    "score_total": -1,
                    "score_breakdown": {
                        "core_intent": 0,
                        "implicit_signal": 0,
                        "target_or_social_meaning": 0,
                        "hallucination_penalty": 0,
                        "literal_only_penalty": 0
                    },
                    "verdict": "wrong",
                    "reasoning_short": "Inference error."
                }
            })
            existing.add((video_id, model_name))
            print(f"[ok] {video_id} | {model_name} | inference error handled")
            continue

        user_text = f"""
Question:
{qa["question"]}

Gold Answer:
{qa["gold_answer"]}

Literal Summary:
{qa["literal_summary"]}

Intended Meaning:
{qa["intended_meaning"]}

Evidence:
{json.dumps(qa["evidence"], ensure_ascii=False)}

Rubric:
{json.dumps(qa["rubric"], ensure_ascii=False)}

Model Prediction:
{prediction}

Judge this prediction.
"""

        try:
            judgment = call_structured_response(
                model=JUDGE_MODEL,
                system_prompt=JUDGE_SYSTEM_PROMPT,
                user_text=user_text,
                image_paths=None,
                schema_name="oe_judgment",
                schema=JUDGE_SCHEMA
            )

            append_jsonl(OUT_PATH, {
                "video_id": video_id,
                "model_name": model_name,
                "model_family": row.get("model_family", ""),
                "size_bucket": row.get("size_bucket", ""),
                "open_closed": row.get("open_closed", ""),
                "provider": row.get("provider", ""),
                "question": qa["question"],
                "gold_answer": qa["gold_answer"],
                "literal_summary": qa["literal_summary"],
                "intended_meaning": qa["intended_meaning"],
                "evidence": qa["evidence"],
                "prediction": prediction,
                "judgment": judgment,
            })

            existing.add((video_id, model_name))
            print(f"[ok] {video_id} | {model_name} | score={judgment['score_total']}")

        except Exception as e:
            print(f"[error] {video_id} | {model_name} | {e}")


if __name__ == "__main__":
    main()