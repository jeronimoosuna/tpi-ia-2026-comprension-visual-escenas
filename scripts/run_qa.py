import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import qa_engine, tracing

ROOT = Path(__file__).resolve().parents[1]
DEVELOPMENT_DIR = ROOT / "data" / "development"
QUESTIONS_PATH = ROOT / "data" / "tpi_2026_qa_public" / "questions_dev.jsonl"
SCENES_PATH = ROOT / "reports" / "scene_json" / "scenes_enriched.json"
OUTPUT_DIR = ROOT / "reports" / "qa"
TRACE_PATH = ROOT / "reports" / "trace" / "pipeline.jsonl"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tracer = tracing.Tracer(TRACE_PATH)
    annotations = json.load(open(DEVELOPMENT_DIR / "annotations.json"))
    categories = annotations["categories"]
    scenes = json.load(open(SCENES_PATH))
    questions = [json.loads(line) for line in open(QUESTIONS_PATH)]

    gt_counts = Counter((a["image_id"], a["category_id"]) for a in annotations["annotations"])
    name_to_id = {c["name"]: c["id"] for c in categories}

    answers = []
    count_correct = count_total = 0
    exist_correct = exist_total = 0

    for q in questions:
        image_id = q["image_file"].rsplit(".", 1)[0]
        scene = scenes[image_id]
        with tracer.step(image_id, "qa_engine", {"question_id": q["question_id"], "type": q["type"]}):
            answer = qa_engine.answer_question(scene, q, categories)
        answers.append(answer)

        if q["type"] in ("count", "existence") and q.get("class_en"):
            cat_id = name_to_id.get(q["class_en"][0])
            gt_n = gt_counts.get((q["image_id"], cat_id), 0) if cat_id is not None else 0
            if q["type"] == "count":
                count_total += 1
                count_correct += int(answer["normalized_answer"] == gt_n)
            else:
                exist_total += 1
                gt_yn = "yes" if gt_n > 0 else "no"
                exist_correct += int(answer["normalized_answer"] == gt_yn)

    with open(OUTPUT_DIR / "answers_dev.json", "w", encoding="utf-8") as f:
        json.dump(answers, f, ensure_ascii=False, indent=2)

    by_type = Counter(a["type"] for a in answers)
    unsupported_by_type = Counter(a["type"] for a in answers if a["normalized_answer"] == "unsupported")

    summary = {
        "num_questions": len(answers),
        "by_type": dict(by_type),
        "unsupported_by_type": dict(unsupported_by_type),
        "count_exact_match": count_correct,
        "count_total": count_total,
        "existence_accuracy_correct": exist_correct,
        "existence_total": exist_total,
    }
    with open(OUTPUT_DIR / "qa_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Questions answered: {len(answers)}")
    print(f"By type: {dict(by_type)}")
    print(f"Unsupported by type: {dict(unsupported_by_type)}")
    print(f"Count exact-match: {count_correct}/{count_total}")
    print(f"Existence accuracy: {exist_correct}/{exist_total}")


if __name__ == "__main__":
    main()
