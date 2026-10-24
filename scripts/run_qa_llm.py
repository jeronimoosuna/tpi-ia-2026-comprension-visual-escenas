import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import qa_llm, tracing

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
    name_to_id = {c["name"]: c["id"] for c in categories}
    scenes = json.load(open(SCENES_PATH))
    questions = [json.loads(line) for line in open(QUESTIONS_PATH)]

    answers = []
    t_start = time.time()
    class_match = class_total = 0
    for i, q in enumerate(questions):
        image_id = q["image_file"].rsplit(".", 1)[0]
        scene = scenes[image_id]
        t0 = time.time()
        with tracer.step(image_id, "qa_llm", {"question_id": q["question_id"], "type": q["type"]}):
            answer = qa_llm.answer_question_llm(scene, q, categories)
        elapsed = time.time() - t0
        answers.append(answer)

        gt_classes = set(q.get("class_en", []))
        if gt_classes:
            class_total += 1
            pred_classes = set(answer["interpreted_query"].get("class_en", []))
            class_match += int(gt_classes == pred_classes)

        print(f"[{i + 1}/{len(questions)}] {q['question_id']} ({q['type']}, {elapsed:.1f}s): "
              f"{answer['normalized_answer']!r}", flush=True)

    total_elapsed = time.time() - t_start

    with open(OUTPUT_DIR / "answers_llm_dev.json", "w", encoding="utf-8") as f:
        json.dump(answers, f, ensure_ascii=False, indent=2)

    by_type = Counter(a["type"] for a in answers)
    unsupported_by_type = Counter(a["type"] for a in answers if a["normalized_answer"] == "unsupported")
    type_correct = sum(
        1 for q, a in zip(questions, answers) if a["interpreted_query"].get("type") == q["type"]
    )

    summary = {
        "model": "Qwen/Qwen2.5-1.5B-Instruct",
        "num_questions": len(answers),
        "total_time_seconds": total_elapsed,
        "by_type": dict(by_type),
        "unsupported_by_type": dict(unsupported_by_type),
        "interpreted_type_correct": type_correct,
        "interpreted_class_en_exact_match": class_match,
        "interpreted_class_en_total": class_total,
    }
    with open(OUTPUT_DIR / "qa_llm_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print()
    print(f"Total: {total_elapsed:.1f}s")
    print(f"By type: {dict(by_type)}")
    print(f"Unsupported by type: {dict(unsupported_by_type)}")
    print(f"interpreted type correct: {type_correct}/{len(questions)}")
    print(f"interpreted class_en exact match: {class_match}/{class_total}")


if __name__ == "__main__":
    main()
