
import torch

import warnings
from tqdm import tqdm
import numpy as np
from sklearn.metrics import pairwise_distances

from .llm_models import get_transformer_llm

def compute_doc_embedding(text, *, model=None, tokenizer=None, model_name='SCINCL', device='cpu', max_length=512):
    use_gpu = True  
    if device == 'cpu': 
        use_gpu = False

    if model is None and tokenizer is None:
        tokenizer, model, device = get_transformer_llm(model_name, use_gpu)
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt", max_length=max_length).to(device)
    result = model(**inputs)
    
    return result.last_hidden_state[0].mean(0).detach().cpu().numpy()
    # if not use_gpu:
    #     return torch.mean(result.last_hidden_state[0], dim=0).detach().numpy()
    # else:
    #     return torch.mean(result.last_hidden_state[0], dim=0).cpu().detach().numpy()

def compute_embeddings(df, *, model_name='SCINCL', cols=['title', 'abstract'], sep_token='[SEP]', as_np=False, use_gpu=True):
    tokenizer, model, device = get_transformer_llm(model_name, use_gpu)
    if use_gpu and device == 'cpu':
        warnings.warn(f'Tried to use GPU, but GPU is not available. Using {device}.')

    papers = df[cols].fillna('').apply(lambda row: sep_token.join(row), axis=1).reset_index()
    papers.columns = ['id', 'text']

    embeddings = {}
    for _, row in tqdm(papers.iterrows(), total=len(papers)):
        embedding = compute_doc_embedding(row['text'], model=model, tokenizer=tokenizer, device=device)
        embeddings[row['id']] = embedding

    if as_np:
        return np.array(list(embeddings.values()))
    return embeddings

def closest_embedding_to_centroid(embeddings, centroid, metric='cosine'):
    embeddings_array = np.array(list(embeddings.values()))
    distances = pairwise_distances(embeddings_array, [centroid], metric=metric)
    min_index = np.argmin(distances)
    return min_index, centroid

def compute_centroids(embeddings_dict, df):
    """
    Just like the snippet: compute mean embedding per cluster.
    """
    centroids = {}
    for cluster_id in df['cluster'].unique():
        idxs = df[df['cluster'] == cluster_id].index
        embs = [embeddings_dict[i] for i in idxs if i in embeddings_dict]
        if embs:
            centroids[cluster_id] = np.mean(embs, axis=0)
        else:
            centroids[cluster_id] = None
    return centroids
