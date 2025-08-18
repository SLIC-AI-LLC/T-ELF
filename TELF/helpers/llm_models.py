import subprocess
from openai import OpenAI
import subprocess
from transformers import AutoTokenizer, AutoModel
import torch
from langchain_ollama import OllamaLLM

def get_ollama_llm(model: str, base_url: str, temperature: float) -> OllamaLLM:
    """
    Create and return a configured OllamaLLM instance.
    If `model` isn’t yet pulled locally, this will shell out to:
        ollama pull <model>
    """
    try:
        available = subprocess.check_output(
            ["ollama", "list"], text=True, stderr=subprocess.DEVNULL
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Could not list Ollama models: {e}")

    if model not in available:
        try:
            print(f"Model '{model}' not found locally – pulling…")
            subprocess.run(
                ["ollama", "pull", model],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to pull Ollama model '{model}': {e.stderr.strip()}")

    return OllamaLLM(model=model, base_url=base_url, temperature=temperature)

def get_openai_llm(openai_api_key:str):
    client = OpenAI(api_key=openai_api_key)
    return client

def get_model_name(model_name):
    assert model_name in ('SCINCL', 'SPECTER'), f'Unknown model name "{model_name}"!'
    if model_name == 'SCINCL':
        return 'malteos/scincl'
    if model_name == 'SPECTER':
        return 'allenai/specter2_base'
    return model_name

def get_transformer_llm(embedding_model: str, device=False):
    if isinstance(device, int) and torch.cuda.is_available():
        device = f'cuda:{device}'
        # this is hackish, needs a fix, but im not doing it right now
        if 'false' in device.lower():
            device = 'cpu'
        elif 'true' in device.lower():
            device = 'cuda:0'
    else:
        device = 'cuda:0' if device and torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')
    tokenizer = AutoTokenizer.from_pretrained(get_model_name(embedding_model))
    model = AutoModel.from_pretrained(get_model_name(embedding_model)).to(device)
    return tokenizer, model, device