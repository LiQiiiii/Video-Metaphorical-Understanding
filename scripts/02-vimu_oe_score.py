import os
import json
from collections import defaultdict


PROJECT_ROOT = "/Your/Path/To/ViMU"

INPUT_PATH = os.path.join(PROJECT_ROOT, "output", "vimu_oe_judged.jsonl")

ALL_JSONL = os.path.join(
    PROJECT_ROOT, "output", "vimu_oe_judged_scored.jsonl"
)

STATS_JSON = os.path.join(
    PROJECT_ROOT, "output", "vimu_oe_summary.json"
)

MIN_SCORE = -6
MAX_SCORE = 9
RANGE = MAX_SCORE - MIN_SCORE


def load_jsonl(path):
    if not os.path.exists(path):
        return []

    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def normalize(score: float) -> float:
    return (score - MIN_SCORE) / RANGE * 100.0


def get_score(row):
    return row.get("judgment", {}).get("score_total", None)


def main():
    rows = load_jsonl(INPUT_PATH)

    rows = [
        r for r in rows
        if "video_id" in r
        and "model_name" in r
        and get_score(r) is not None
    ]

    by_video = defaultdict(dict)
    all_models = set()

    for r in rows:
        by_video[r["video_id"]][r["model_name"]] = r
        all_models.add(r["model_name"])

    all_models = sorted(all_models)
    num_models = len(all_models)

    complete_video_ids = sorted([
        video_id
        for video_id, model_map in by_video.items()
        if len(model_map) == num_models
    ])
    complete_video_id_set = set(complete_video_ids)

    complete_rows = [
        r for r in rows
        if r["video_id"] in complete_video_id_set
    ]

    write_jsonl(ALL_JSONL, complete_rows)

    model_scores = defaultdict(list)

    for r in complete_rows:
        model_scores[r["model_name"]].append(get_score(r))

    stats = []

    for model_name in all_models:
        scores = model_scores.get(model_name, [])
        if not scores:
            continue

        avg_raw = sum(scores) / len(scores)

        stats.append({
            "model_name": model_name,
            "num_samples": len(scores),
            "avg_raw_score": avg_raw,
            "avg_normalized_score_100": normalize(avg_raw),
            "success_rate_score_ge_3": sum(s >= 3 for s in scores) / len(scores),
            "perfect_rate_score_eq_9": sum(s == 9 for s in scores) / len(scores),
            "wrong_rate_score_le_0": sum(s <= 0 for s in scores) / len(scores),
        })

    stats = sorted(
        stats,
        key=lambda x: x["avg_normalized_score_100"],
        reverse=True
    )

    write_json(STATS_JSON, {
        "input_path": INPUT_PATH,
        "num_models": num_models,
        "num_complete_videos": len(complete_video_ids),
        "complete_judgment_count": len(complete_rows),
        "model_stats": stats,
    })

    print(f"[ok] saved complete judged rows to: {ALL_JSONL}")
    print(f"[ok] saved model stats to: {STATS_JSON}")

if __name__ == "__main__":
    main()