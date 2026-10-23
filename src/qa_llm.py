def interpret_query(question: str) -> dict:
    """Receives only the question text; returns an interpreted_query JSON
    (intent/operation, target objects or filters, expected output type)
    to be validated and executed by qa_engine.answer_question."""
    raise NotImplementedError


def generate_description_llm(scene_json: dict) -> str:
    """Generates the factual description from scene_json only, never from the image."""
    raise NotImplementedError
