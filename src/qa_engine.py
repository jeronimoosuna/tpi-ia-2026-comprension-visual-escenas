import re
from collections import Counter

from src.geometry import bbox_iou, bbox_to_corners
from src.properties import OVERLAP_IOU_THRESHOLD

QUESTION_TYPES = ["description", "count", "existence", "position", "size", "relation"]

# Deterministic mode receives the question text too, but must not use an LLM
# to interpret it. For "relation" questions the predicate (above/left_of/...)
# is not a separate structured field from the cátedra -- only class_en/
# class_es (the objects) and type are. These are fixed-phrase keyword
# patterns over the known Spanish wording, not free-form NLU: reading a
# known pattern by code, not "interpreting" the question.
_PREDICATE_PATTERNS = [
    (re.compile(r"por encima de|arriba de", re.IGNORECASE), "above"),
    (re.compile(r"por debajo de|abajo de", re.IGNORECASE), "below"),
    (re.compile(r"a la izquierda de", re.IGNORECASE), "left_of"),
    (re.compile(r"a la derecha de", re.IGNORECASE), "right_of"),
    (re.compile(r"se superpone|superposici[oó]n|solapa", re.IGNORECASE), "overlaps"),
]


def _extract_predicate(question_text: str) -> str | None:
    for pattern, predicate in _PREDICATE_PATTERNS:
        if pattern.search(question_text):
            return predicate
    return None


def _find_by_category(scene_json: dict, category_id: int) -> list[dict]:
    return [o for o in scene_json["objects"] if o["category_id"] == category_id]


def _base_answer(question_record: dict) -> dict:
    return {
        "question_id": question_record["question_id"],
        "type": question_record["type"],
        "question": question_record["question"],
    }


def _unsupported(question_record: dict) -> dict:
    return {
        **_base_answer(question_record),
        "answer": "unsupported",
        "normalized_answer": "unsupported",
        "evidence": [],
        "confidence": 0.0,
    }


def _resolve_single_category(question_record: dict, name_to_id: dict[str, int], index: int = 0) -> int | None:
    names = question_record.get("class_en", [])
    if len(names) <= index or names[index] not in name_to_id:
        return None
    return name_to_id[names[index]]


def _answer_count(scene_json: dict, q: dict, name_to_id: dict[str, int]) -> dict:
    category_id = _resolve_single_category(q, name_to_id)
    if category_id is None:
        return _unsupported(q)
    matches = _find_by_category(scene_json, category_id)
    n = len(matches)
    confidence = min((o["confidence"] for o in matches), default=1.0)
    return {
        **_base_answer(q),
        "answer": str(n),
        "normalized_answer": n,
        "evidence": [o["id"] for o in matches],
        "confidence": confidence,
    }


def _answer_existence(scene_json: dict, q: dict, name_to_id: dict[str, int]) -> dict:
    category_id = _resolve_single_category(q, name_to_id)
    if category_id is None:
        return _unsupported(q)
    matches = _find_by_category(scene_json, category_id)
    exists = len(matches) > 0
    confidence = min((o["confidence"] for o in matches), default=1.0)
    normalized = "yes" if exists else "no"
    return {
        **_base_answer(q),
        "answer": normalized,
        "normalized_answer": normalized,
        "evidence": [o["id"] for o in matches],
        "confidence": confidence,
    }


def _answer_single_instance_field(scene_json: dict, q: dict, name_to_id: dict[str, int], field: str) -> dict:
    """Shared logic for "position"/"size": the question presupposes exactly
    one instance of the category ("la instancia de X"). Zero matches means
    the question doesn't apply; more than one is ambiguous (we can't know
    which instance is meant) -- both cases return unsupported rather than
    guessing."""
    category_id = _resolve_single_category(q, name_to_id)
    if category_id is None:
        return _unsupported(q)
    matches = _find_by_category(scene_json, category_id)
    if len(matches) != 1:
        return _unsupported(q)
    obj = matches[0]
    return {
        **_base_answer(q),
        "answer": obj[field],
        "normalized_answer": obj[field],
        "evidence": [obj["id"]],
        "confidence": obj["confidence"],
    }


def _evaluate_predicate(subject: dict, obj: dict, predicate: str) -> bool:
    if predicate == "overlaps":
        return bbox_iou(subject["bbox"], obj["bbox"]) > OVERLAP_IOU_THRESHOLD

    sx1, sy1, sx2, sy2 = bbox_to_corners(subject["bbox"])
    ox1, oy1, ox2, oy2 = bbox_to_corners(obj["bbox"])
    scx, scy = (sx1 + sx2) / 2, (sy1 + sy2) / 2
    ocx, ocy = (ox1 + ox2) / 2, (oy1 + oy2) / 2

    if predicate == "left_of":
        return scx < ocx
    if predicate == "right_of":
        return scx > ocx
    if predicate == "above":
        return scy < ocy
    if predicate == "below":
        return scy > ocy
    return False


def _answer_relation(scene_json: dict, q: dict, name_to_id: dict[str, int]) -> dict:
    subject_category = _resolve_single_category(q, name_to_id, index=0)
    object_category = _resolve_single_category(q, name_to_id, index=1)
    if subject_category is None or object_category is None:
        return _unsupported(q)

    subject_matches = _find_by_category(scene_json, subject_category)
    object_matches = _find_by_category(scene_json, object_category)
    if len(subject_matches) != 1 or len(object_matches) != 1:
        return _unsupported(q)

    predicate = _extract_predicate(q["question"])
    if predicate is None:
        return _unsupported(q)

    subject, obj = subject_matches[0], object_matches[0]
    holds = _evaluate_predicate(subject, obj, predicate)
    normalized = "yes" if holds else "no"
    return {
        **_base_answer(q),
        "answer": normalized,
        "normalized_answer": normalized,
        "evidence": [subject["id"], obj["id"]],
        "confidence": min(subject["confidence"], obj["confidence"]),
    }


def _answer_description(scene_json: dict, q: dict, categories: list[dict]) -> dict:
    text = describe_scene(scene_json, categories)
    objects = scene_json["objects"]
    return {
        **_base_answer(q),
        "answer": text,
        "normalized_answer": text,
        "evidence": [o["id"] for o in objects],
        "confidence": min((o["confidence"] for o in objects), default=1.0),
    }


def answer_question(scene_json: dict, question_record: dict, categories: list[dict]) -> dict:
    """question_record: one row as provided by the cátedra (question_id,
    type, question, class_en, ...), e.g. from questions_dev.jsonl.
    categories: [{"id": int, "name": str}, ...] (same list as
    annotations.json["categories"]), used to resolve class_en to
    category_id. Returns a questionAnswer dict (question_id, type,
    question, answer, normalized_answer, evidence, confidence), with
    normalized_answer="unsupported" when the query can't be resolved
    from scene_json alone."""
    name_to_id = {c["name"]: c["id"] for c in categories}
    qtype = question_record["type"]

    if qtype == "description":
        return _answer_description(scene_json, question_record, categories)
    if qtype == "count":
        return _answer_count(scene_json, question_record, name_to_id)
    if qtype == "existence":
        return _answer_existence(scene_json, question_record, name_to_id)
    if qtype == "position":
        return _answer_single_instance_field(scene_json, question_record, name_to_id, "position")
    if qtype == "size":
        return _answer_single_instance_field(scene_json, question_record, name_to_id, "size")
    if qtype == "relation":
        return _answer_relation(scene_json, question_record, name_to_id)
    return _unsupported(question_record)


def describe_scene(scene_json: dict, categories: list[dict]) -> str:
    """Deterministic, template-based description (no LLM): the top-5 most
    frequent categories in the scene with their counts, in Spanish."""
    id_to_name = {c["id"]: c["name"] for c in categories}
    counts = Counter(o["category_id"] for o in scene_json["objects"])
    if not counts:
        return "No se detectaron objetos en la escena."

    top = counts.most_common(5)
    parts = [f"{n} {id_to_name.get(cid, f'categoria_{cid}')}" for cid, n in top]
    if len(parts) == 1:
        return f"La escena contiene {parts[0]}."
    return f"La escena contiene {', '.join(parts[:-1])} y {parts[-1]}."
