import json
import re
from collections import Counter

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src import qa_engine
from src.properties import PREDICATES

_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
_tokenizer = None
_model = None

_INTERPRET_SYSTEM_PROMPT = """Convertís una pregunta en español sobre una escena a un JSON de consulta estructurada. Respondé UNICAMENTE con el JSON en una sola linea, sin texto adicional, sin markdown.

Campos:
- "type": uno de ["description", "count", "existence", "position", "size", "relation"]
- "class_en": lista de nombres de categoria EN INGLES, copiados EXACTAMENTE de esta lista (nunca traduzcas vos, buscá el nombre que mas se parezca semánticamente): {categories}
- "predicate": solo si type es "relation", uno de ["left_of", "right_of", "above", "below", "overlaps"]

Ejemplos:
P: "Describí brevemente la escena."
{{"type": "description", "class_en": []}}

P: "¿Cuántas instancias de la categoría pomo de puerta aparecen en la imagen?"
{{"type": "count", "class_en": ["doorknob"]}}

P: "¿Hay al menos una instancia de tomacorriente en la imagen?"
{{"type": "existence", "class_en": ["wall socket"]}}

P: "¿En qué zona de la imagen está la instancia de teléfono celular?"
{{"type": "position", "class_en": ["cell phone"]}}

P: "¿Qué tamaño relativo tiene la instancia de collar?"
{{"type": "size", "class_en": ["necklace"]}}

P: "¿La instancia de reloj está por encima de la instancia de anillo?"
{{"type": "relation", "class_en": ["clock", "ring"], "predicate": "above"}}
"""

_DESCRIPTION_SYSTEM_PROMPT = (
    "Escribís una descripción breve y factual de una escena en español, usando "
    "EXCLUSIVAMENTE la lista de objetos detectados que te paso (categoría y "
    "cantidad). No inventes objetos, cantidades, colores ni ningún detalle que "
    "no esté en la lista. Respondé solo con la descripción, 1 o 2 oraciones, "
    "sin texto adicional."
)


def _get_model():
    global _tokenizer, _model
    if _model is None:
        _tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME)
        _model = AutoModelForCausalLM.from_pretrained(_MODEL_NAME)
        _model.eval()
    return _tokenizer, _model


def _generate(system_prompt: str, user_prompt: str, max_new_tokens: int) -> str:
    tokenizer, model = _get_model()
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def _extract_json(text: str) -> dict:
    """Finds the first {...} block anywhere in the output (the model
    sometimes wraps it in markdown fences or prose) rather than assuming
    the whole output is a bare single-line JSON object."""
    match = re.search(r"\{.*?\}", text, re.DOTALL)
    if not match:
        return {"type": None}
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {"type": None}
    return parsed if isinstance(parsed, dict) else {"type": None}


def interpret_query(question: str, categories: list[dict]) -> dict:
    """Receives only the question text (no type/class_en/class_es from the
    cátedra's structured fields -- those are reserved for deterministic
    mode). Returns the intermediate JSON {type, class_en, [predicate]},
    produced by the LLM, not validated yet (answer_question_llm validates
    it against the known category vocabulary before executing it)."""
    category_names = ", ".join(sorted(c["name"] for c in categories))
    system_prompt = _INTERPRET_SYSTEM_PROMPT.format(categories=category_names)
    raw = _generate(system_prompt, f'P: "{question}"', max_new_tokens=80)
    return _extract_json(raw)


def generate_description_llm(scene_json: dict, categories: list[dict]) -> str:
    """Generates the factual description from scene_json only, never from
    the image. The prompt only gets category+count pairs (not the raw
    object list, which could be hundreds of rows on a crowded image) with
    an explicit instruction not to invent anything beyond that list."""
    id_to_name = {c["id"]: c["name"] for c in categories}
    counts = Counter(o["category_id"] for o in scene_json["objects"])
    if not counts:
        return "No se detectaron objetos en la escena."

    top = counts.most_common(8)
    object_list = ", ".join(f"{n} {id_to_name.get(cid, 'objeto')}" for cid, n in top)
    user_prompt = f"Objetos detectados (categoría y cantidad): {object_list}"
    return _generate(_DESCRIPTION_SYSTEM_PROMPT, user_prompt, max_new_tokens=120)


def _is_valid_interpretation(interpreted: dict, name_to_id: dict[str, int]) -> bool:
    if interpreted.get("type") not in qa_engine.QUESTION_TYPES:
        return False
    names = interpreted.get("class_en", [])
    if not isinstance(names, list) or not all(isinstance(n, str) and n in name_to_id for n in names):
        return False
    if interpreted["type"] == "relation" and interpreted.get("predicate") not in PREDICATES:
        return False
    return True


def answer_question_llm(scene_json: dict, question_record: dict, categories: list[dict]) -> dict:
    """question_record: only question_id/type/question are read from it for
    bookkeeping (preserved in the output, same as deterministic mode) --
    type/class_en/class_es are never passed to the LLM, only question_record
    ["question"]. Returns a questionAnswerLLM dict: everything
    answer_question returns, plus "interpreted_query" with the LLM's raw
    intermediate JSON (even when invalid/unsupported, for auditability)."""
    name_to_id = {c["name"]: c["id"] for c in categories}
    question_text = question_record["question"]
    interpreted = interpret_query(question_text, categories)

    if interpreted.get("type") == "description":
        text = generate_description_llm(scene_json, categories)
        objects = scene_json["objects"]
        result = {
            "question_id": question_record["question_id"],
            "type": question_record["type"],
            "question": question_text,
            "answer": text,
            "normalized_answer": text,
            "evidence": [o["id"] for o in objects],
            "confidence": min((o["confidence"] for o in objects), default=1.0),
        }
    elif _is_valid_interpretation(interpreted, name_to_id):
        synthetic_record = {
            "question_id": question_record["question_id"],
            "type": interpreted["type"],
            "question": question_text,
            "class_en": interpreted.get("class_en", []),
        }
        if "predicate" in interpreted:
            synthetic_record["predicate"] = interpreted["predicate"]
        result = qa_engine.answer_question(scene_json, synthetic_record, categories)
        result["question_id"] = question_record["question_id"]
        result["question"] = question_text
    else:
        result = qa_engine.unsupported_answer(question_record)

    result["interpreted_query"] = interpreted
    return result
