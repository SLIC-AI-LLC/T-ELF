from typing import Iterable

def build_json_vote_prompt(candidate: str, contexts: Iterable[str]) -> str:
    """
    Build a JSON-only prompt from example contexts and a candidate string.
    """
    ctx_block = "\n----\n".join(contexts)
    return (
        "You are an expert researcher. Output ONLY valid JSON.\n"
        f"Target context examples:\n{ctx_block}\n\n"
        f"Candidate abstract:\n{candidate}\n"
        "Given the context, is the candidate about any of the concepts? "
        'Respond {"answer":"yes|no","reason":"..."}'
    )

def produce_label_prompt(words:Iterable[str]):
    prompt = (
        f"Use the following words to establish a theme or a topic: {', '.join(words)!r}. "
        "The output should be a label that has at most 3 tokens and is characterized by the provided words. "
        "The output should not be in quotes."
    )
    return prompt
