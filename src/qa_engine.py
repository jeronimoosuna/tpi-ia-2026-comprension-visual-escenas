QUESTION_TYPES = ["description", "count", "existence", "position", "size", "relation"]


def answer_question(scene_json: dict, question_id: str, question: str, structured_query: dict) -> dict:
    """structured_query carries the operation/target fields; returns a questionAnswer dict
    (question_id, type, question, answer, normalized_answer, evidence, confidence),
    or normalized_answer="unsupported" if the query can't be resolved from scene_json."""
    raise NotImplementedError


def describe_scene(scene_json: dict) -> str:
    raise NotImplementedError
