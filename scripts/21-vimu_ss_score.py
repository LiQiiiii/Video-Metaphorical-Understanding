import os
import json
import argparse
from collections import defaultdict

from scripts.utils import load_jsonl, append_jsonl, ensure_dir


PROJECT_ROOT = "/Your/Path/To/ViMU"

MCQ_PATH = os.path.join(PROJECT_ROOT, "metadata", "vimu_ss.jsonl")


def score_multiselect(pred: list[str], gt: list[str]) -> float:
    pred_set = set(pred)
    gt_set = set(gt)

    # Any false positive => 0
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


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt_mode",
        type=str,
        default="with_guidance",
        choices=["with_guidance", "without_guidance"],
        help="Score predictions generated with or without taxonomy guidance."
    )

    parser.add_argument(
        "--pred_path",
        type=str,
        default=None,
        help="Optional prediction path. If not set, inferred from prompt_mode."
    )

    parser.add_argument(
        "--out_path",
        type=str,
        default=None,
        help="Optional scored output path. If not set, inferred from prompt_mode."
    )

    parser.add_argument(
        "--summary_path",
        type=str,
        default=None,
        help="Optional summary output path. If not set, inferred from prompt_mode."
    )

    return parser.parse_args()


def resolve_paths(args):
    suffix = args.prompt_mode

    pred_path = args.pred_path or os.path.join(
        PROJECT_ROOT,
        "output",
        f"vimu_ss_predictions_{suffix}.jsonl"
    )

    out_path = args.out_path or os.path.join(
        PROJECT_ROOT,
        "output",
        f"vimu_ss_{suffix}_scored.jsonl"
    )

    summary_path = args.summary_path or os.path.join(
        PROJECT_ROOT,
        "output",
        f"vimu_ss_{suffix}_summary.json"
    )

    return pred_path, out_path, summary_path


def main():
    args = parse_args()
    pred_path, out_path, summary_path = resolve_paths(args)

    ensure_dir(os.path.dirname(out_path))
    ensure_dir(os.path.dirname(summary_path))

    mcq_rows = load_jsonl(MCQ_PATH)
    pred_rows = load_jsonl(pred_path)

    mcq_map = {
        r["video_id"]: r
        for r in mcq_rows
        if "video_id" in r and "mcq_tasks" in r
    }

    existing = {
        (r["video_id"], r["model_name"], r["task_name"])
        for r in load_jsonl(out_path)
        if "video_id" in r and "model_name" in r and "task_name" in r
    }

    print(f"Loaded {len(mcq_rows)} SS MCQ rows.")
    print(f"Loaded {len(pred_rows)} prediction rows from: {pred_path}")
    print(f"Loaded {len(existing)} existing scored rows.")
    print(f"Prompt mode: {args.prompt_mode}")

    for row in pred_rows:
        video_id = row["video_id"]
        model_name = row["model_name"]
        task_name = row["task_name"]

        key = (video_id, model_name, task_name)
        if key in existing:
            continue

        if video_id not in mcq_map:
            print(f"[skip] no MCQ row for {video_id}")
            continue

        task_block = mcq_map[video_id]["mcq_tasks"].get(task_name)

        if task_block is None:
            print(f"[skip] no task block for {video_id} | {task_name}")
            continue

        gt = normalize_list(task_block.get("ground_truth", []))
        pred = normalize_list(
            row.get("prediction", {}).get("selected_options", [])
        )

        score = score_multiselect(pred, gt)

        append_jsonl(out_path, {
            "video_id": video_id,
            "model_name": model_name,
            "task_name": task_name,
            "ground_truth": gt,
            "ground_truth_labels": task_block.get("ground_truth_labels", []),
            "prediction": pred,
            "score": score,
            "prompt_mode": args.prompt_mode,
        })

        existing.add(key)

        print(
            f"[scored] {video_id} | {model_name} | {task_name} | "
            f"pred={pred} | gt={gt} | score={score:.3f}"
        )

    scored_rows = load_jsonl(out_path)

    model_task_scores = defaultdict(list)

    for r in scored_rows:
        model_task_scores[(r["model_name"], r["task_name"])].append(r["score"])

    by_task = []

    for (model_name, task_name), scores in model_task_scores.items():
        avg_score = sum(scores) / len(scores) if scores else 0.0

        by_task.append({
            "model_name": model_name,
            "task_name": task_name,
            "num_samples": len(scores),
            "avg_score_0_1": avg_score,
            "avg_score_100": avg_score * 100.0,
            "exact_match_rate": sum(s == 1.0 for s in scores) / len(scores) if scores else 0.0,
            "zero_score_rate": sum(s == 0.0 for s in scores) / len(scores) if scores else 0.0,
        })

    by_task = sorted(
        by_task,
        key=lambda x: (x["task_name"], -x["avg_score_100"])
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
            "exact_match_rate": sum(s == 1.0 for s in scores) / len(scores) if scores else 0.0,
            "zero_score_rate": sum(s == 0.0 for s in scores) / len(scores) if scores else 0.0,
        })

    overall = sorted(
        overall,
        key=lambda x: x["avg_score_100"],
        reverse=True
    )

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "prompt_mode": args.prompt_mode,
            "prediction_path": pred_path,
            "scored_path": out_path,
            "score_rule": "If prediction contains any wrong option, score = 0. Otherwise score = |pred ∩ gt| / |gt|.",
            "num_mcq_rows": len(mcq_rows),
            "num_prediction_rows": len(pred_rows),
            "num_scored_rows": len(scored_rows),
            "by_task": by_task,
            "overall": overall,
        }, f, ensure_ascii=False, indent=2)

    print(f"\n[ok] saved scored rows to {out_path}")
    print(f"[ok] saved summary to {summary_path}")


if __name__ == "__main__":
    main()