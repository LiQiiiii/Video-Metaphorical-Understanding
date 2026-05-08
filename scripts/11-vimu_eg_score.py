import os
import json
from collections import defaultdict

from scripts.utils import load_jsonl, append_jsonl, ensure_dir


PROJECT_ROOT = "/Your/Path/To/ViMU"

QUESTION_PATH = os.path.join(PROJECT_ROOT, "metadata", "vimu_eg.jsonl")
PRED_PATH = os.path.join(PROJECT_ROOT, "output", "vimu_eg_predictions.jsonl")
OUT_PATH = os.path.join(PROJECT_ROOT, "output", "vimu_eg_scored.jsonl")
SUMMARY_PATH = os.path.join(PROJECT_ROOT, "output", "vimu_eg_summary.json")


def score_multiselect(pred: list[str], gt: list[str]) -> float:
    pred_set = set(pred)
    gt_set = set(gt)

    if pred_set - gt_set:
        return 0.0

    if not gt_set:
        return 0.0

    return len(pred_set & gt_set) / len(gt_set)


def normalize_list(xs) -> list[str]:
    if not isinstance(xs, list):
        return []

    out = []
    seen = set()

    for x in xs:
        x = str(x).strip()
        if not x:
            continue

        if x not in seen:
            seen.add(x)
            out.append(x)

    return out


def get_ground_truth(row):
    if "correct_options" in row:
        return normalize_list(row["correct_options"])

    if "ground_truth" in row:
        return normalize_list(row["ground_truth"])

    if "answer" in row:
        return normalize_list(row["answer"])

    return []


def main():
    ensure_dir(os.path.dirname(OUT_PATH))

    question_rows = load_jsonl(QUESTION_PATH)
    pred_rows = load_jsonl(PRED_PATH)

    question_map = {
        r["video_id"]: r
        for r in question_rows
        if "video_id" in r
    }

    existing = {
        (r["video_id"], r["model_name"], r.get("task_name", "evidence_grounding"))
        for r in load_jsonl(OUT_PATH)
        if "video_id" in r and "model_name" in r
    }

    print(f"Loaded {len(question_rows)} EG question rows.")
    print(f"Loaded {len(pred_rows)} EG prediction rows.")
    print(f"Loaded {len(existing)} existing scored rows.")

    for row in pred_rows:
        video_id = row["video_id"]
        model_name = row["model_name"]
        task_name = row.get("task_name", "evidence_grounding")

        key = (video_id, model_name, task_name)
        if key in existing:
            continue

        if video_id not in question_map:
            print(f"[skip] no EG question for {video_id}")
            continue

        q = question_map[video_id]

        gt = get_ground_truth(q)
        pred = normalize_list(
            row.get("prediction", {}).get("selected_options", [])
        )

        score = score_multiselect(pred, gt)

        append_jsonl(OUT_PATH, {
            "video_id": video_id,
            "model_name": model_name,
            "task_name": task_name,
            "ground_truth": gt,
            "prediction": pred,
            "score": score,
            "score_100": score * 100.0,
        })

        existing.add(key)

        print(
            f"[scored] {video_id} | {model_name} | "
            f"pred={pred} | gt={gt} | score={score:.3f}"
        )

    scored_rows = load_jsonl(OUT_PATH)

    model_task_scores = defaultdict(list)
    model_task_exact = defaultdict(list)
    model_task_pred_count = defaultdict(list)
    model_task_gold_count = defaultdict(list)

    for r in scored_rows:
        model_name = r["model_name"]
        task_name = r.get("task_name", "evidence_grounding")

        pred = normalize_list(r.get("prediction", []))
        gt = normalize_list(r.get("ground_truth", []))

        model_task_scores[(model_name, task_name)].append(r["score"])
        model_task_exact[(model_name, task_name)].append(set(pred) == set(gt))
        model_task_pred_count[(model_name, task_name)].append(len(pred))
        model_task_gold_count[(model_name, task_name)].append(len(gt))

    by_task = []

    for (model_name, task_name), scores in model_task_scores.items():
        avg_score = sum(scores) / len(scores) if scores else 0.0
        exact_rate = (
            sum(model_task_exact[(model_name, task_name)])
            / len(model_task_exact[(model_name, task_name)])
            if model_task_exact[(model_name, task_name)]
            else 0.0
        )
        avg_pred_count = (
            sum(model_task_pred_count[(model_name, task_name)])
            / len(model_task_pred_count[(model_name, task_name)])
            if model_task_pred_count[(model_name, task_name)]
            else 0.0
        )
        avg_gold_count = (
            sum(model_task_gold_count[(model_name, task_name)])
            / len(model_task_gold_count[(model_name, task_name)])
            if model_task_gold_count[(model_name, task_name)]
            else 0.0
        )

        by_task.append({
            "model_name": model_name,
            "task_name": task_name,
            "num_samples": len(scores),
            "avg_score_0_1": avg_score,
            "avg_score_100": avg_score * 100.0,
            "exact_set_match_rate": exact_rate,
            "avg_pred_option_count": avg_pred_count,
            "avg_gold_option_count": avg_gold_count,
            "selection_conservatism": avg_pred_count - avg_gold_count,
        })

    by_task = sorted(
        by_task,
        key=lambda x: x["avg_score_100"],
        reverse=True,
    )

    model_overall = defaultdict(list)

    for (model_name, task_name), scores in model_task_scores.items():
        model_overall[model_name].extend(scores)

    overall = []

    for model_name, scores in model_overall.items():
        avg_score = sum(scores) / len(scores) if scores else 0.0

        overall.append({
            "model_name": model_name,
            "num_samples": len(scores),
            "avg_score_0_1": avg_score,
            "avg_score_100": avg_score * 100.0,
        })

    overall = sorted(
        overall,
        key=lambda x: x["avg_score_100"],
        reverse=True,
    )

    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "scoring_rule": (
                "If prediction contains any option outside the gold set, score = 0; "
                "otherwise score = |pred ∩ gold| / |gold|."
            ),
            "by_task": by_task,
            "overall": overall,
        }, f, ensure_ascii=False, indent=2)

    print(f"[ok] saved scored rows to {OUT_PATH}")
    print(f"[ok] saved summary to {SUMMARY_PATH}")


if __name__ == "__main__":
    main()