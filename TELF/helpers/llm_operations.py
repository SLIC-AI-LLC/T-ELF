import json
import re
import logging
from langchain_ollama import OllamaLLM
import string

from .prompts import produce_label_prompt
log = logging.getLogger(__name__)

def vote_once(llm: OllamaLLM, prompt: str) -> tuple[bool, str]:
    """
    Invoke the given OllamaLLM instance once, strip markdown fences,
    parse its JSON response, and return (yes_flag, reason).
    """
    raw = llm.invoke(prompt)
    txt = re.sub(r"```(?:json)?", "", raw).strip()
    try:
        obj = json.loads(txt)
        yes = obj.get("answer", "").lower() == "yes"
        reason = str(obj.get("reason", "")).strip()
        return yes, reason
    except Exception:
        log.warning("Bad JSON from LLM: %s", raw)
        return False, ""

def produce_label(words: str, client=None, model="gpt-3.5-turbo-instruct"):
    prompt = produce_label_prompt(words)
    
    response = client.completions.create(
        model=model,
        prompt=prompt,
        max_tokens=2048
    )
    label = response.choices[0].text.strip()
    return label.strip().translate(str.maketrans('', '', string.punctuation + '\n'))