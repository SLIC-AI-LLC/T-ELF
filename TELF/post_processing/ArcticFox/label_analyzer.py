"""
LabelAnalyzer – self-contained, patched for your compute_embeddings signature
────────────────────────────────────────────────────────────────────────────
Produces specific labels for:
  1) single string
  2) list of texts
  3) DataFrame one-row per text
  4) DataFrame grouped by cluster_col

Supports both Ollama and OpenAI chat models.

Dependencies
------------
pip install "langchain-community>=0.2.0" "langchain-openai>=0.1.1" \
            pandas numpy scikit-learn tqdm
"""
from __future__ import annotations
import re
from collections import Counter
from typing import Any, Dict, Iterable, List, Mapping, Optional, Union

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise_distances
from tqdm import tqdm

# ── LLM back-ends ───────────────────────────────────────────────────────────
from langchain_community.chat_models import ChatOllama
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from langchain_ollama import OllamaLLM

# ── YOUR embedding helpers ──────────────────────────────────────────────────
from ...helpers.embeddings import (
    compute_doc_embedding,
    compute_embeddings,
    closest_embedding_to_centroid,
    compute_centroids,
)

# ── misc globals ────────────────────────────────────────────────────────────
TOKEN_RE = re.compile(r"[A-Za-z]{3,}")
GENERIC = {
    "analysis","approach","application","model","models","method","methods",
    "research","study","studies","data","system","systems","framework",
    "technology","technologies","network","networks","learning"
}
def _tok(txt: str) -> List[str]:
    return TOKEN_RE.findall(txt.lower())


class LabelAnalyzer:
    def __init__(
        self,
        embedding_model: str = "SCINCL",       # kept for API compatibility
        distance_metric: str   = "cosine",
        text_cols: Optional[List[str]] = None,
        tfidf_top_k: int      = 12,
        include_bigrams: bool = True,
    ):
        self.embedding_model = embedding_model
        self.distance_metric = distance_metric
        self.text_cols       = text_cols or ["title", "abstract"]
        self.tfidf_top_k     = tfidf_top_k
        self.include_bigrams = include_bigrams

    @staticmethod
    def _crit() -> Dict[str, Any]:
        return {
            "minimum words": 2,
            "maximum words": 10,
            "must_not_contain": ["\n", "\r", "  ", "•", "-", ":"]
        }

    def _clean(self, s: str) -> str:
        first = s.splitlines()[0]
        first = re.sub(r"^\s*([-\d]+[.)]|\u2022)\s*", "", first)
        return re.sub(r"\s+", " ", first).strip()

    def _generic(self, s: str) -> bool:
        toks = _tok(s)
        return bool(toks) and all(t in GENERIC for t in toks)

    def _valid(self, s: str, crit: Dict[str,Any]) -> bool:
        if self._generic(s):
            return False
        n = len(s.split())
        if not (crit["minimum words"] <= n <= crit["maximum words"]):
            return False
        if any(b in s for b in crit["must_not_contain"]):
            return False
        return True

    def _distinctive(self, df):
        from collections import Counter
        from sklearn.feature_extraction.text import TfidfVectorizer

        def simple_analyzer(text):
            return text.lower().split()  # replace with tokenizer/lemmatizer if needed

        docs = df["clean_text"].tolist() if "clean_text" in df.columns else df.iloc[:, 0].astype(str).tolist()

        # fallback for very small input
        if len(df) < 2:
            toks = simple_analyzer(" ".join(docs))
            return {0: [w for w, _ in Counter(toks).most_common(self.tfidf_top_k)]}

        try:
            vec = TfidfVectorizer(analyzer=simple_analyzer, min_df=2)
            X = vec.fit_transform(docs)
            vocab = vec.get_feature_names_out()
            cids = df["cluster"].values if "cluster" in df.columns else [0] * len(df)
            return {
                cid: [
                    vocab[i]
                    for i in X[np.array(cids) == cid].sum(axis=0).A1.argsort()[::-1][:self.tfidf_top_k]
                ]
                for cid in sorted(set(cids))
            }
        except ValueError:
            toks = simple_analyzer(" ".join(docs))
            return {0: [w for w, _ in Counter(toks).most_common(self.tfidf_top_k)]}


    def _prompt(self, freq, dist, titles, crit):
        msg = (
            "### Task\nGenerate ONE short and descriptive **research topic label**.\n"
            f"- Use {crit['minimum words']} to {crit['maximum words']} words\n"
            "- Include a distinctive keyword\n"
            "- DO NOT copy any of the titles\n"
            "- Avoid punctuation, numbering, or full sentences\n\n"
            f"**Frequent Keywords:** {', '.join(freq)}\n"
            f"**Distinctive Keywords:** {', '.join(dist)}\n"
            "**Representative Titles:**\n" + "\n".join(f"- {t}" for t in titles) +
            "\nRespond ONLY with a concise label, not a sentence."
        )
        return [
            SystemMessage("You are an expert at naming research topics."),
            HumanMessage(msg)
        ]

    def _cand_ollama(self, freq, dist, titles, crit, model, k):
        chat = ChatOllama(model=model, base_url="http://localhost:11434", temperature=0.6)
        raw = chat.invoke(self._prompt(freq, dist, titles, crit)).content
        parts = [self._clean(p) for p in re.split(r"[\n,]+", raw) if p.strip()]
        
        # Remove any that match titles
        title_set = set(t.lower().strip() for t in titles)
        parts = [p for p in parts if p.lower().strip() not in title_set]
        
        return [p for p in parts if self._valid(p, crit)][:k]


    def _cand_openai(self, freq, dist, titles, crit, model, key, k):
        chat = ChatOpenAI(model=model, api_key=key, temperature=0.6)
        raw  = chat.invoke(self._prompt(freq, dist, titles, crit)).content
        parts= [self._clean(p) for p in re.split(r"[\n,]+", raw) if p.strip()]
        return [p for p in parts if self._valid(p, crit)][:k]

    def _centres(self, df: pd.DataFrame, gpu: bool) -> Dict[int, tuple]:
        emb   = compute_embeddings(df)
        cents = compute_centroids(emb, df)  # ← pass df here
        return {
            cid: closest_embedding_to_centroid(emb, c, metric=self.distance_metric)
            for cid, c in cents.items() if c is not None
        }


    def _pick(self, cands: List[str], centre_vec: np.ndarray, gpu: bool) -> str:
        dev  = "cuda" if gpu else "cpu"
        # ← FIXED: compute_doc_embedding takes only the text
        embs = np.array([compute_doc_embedding(lbl) for lbl in cands])
        d    = pairwise_distances(embs, centre_vec[None, :],
                                  metric=self.distance_metric).ravel()
        return cands[int(d.argmin())]

    def _to_clusters(
        self,
        data: Union[str, Iterable[str], Mapping[Any,str], pd.DataFrame],
        *,
        strat: str = "single",
        col: str   = "cluster",
    ) -> Dict[Any, List[str]]:
        if isinstance(data, str):
            return {0: [data]}
        if isinstance(data, Mapping):
            return {k:[v] for k,v in data.items()}
        if isinstance(data, (list, tuple, pd.Series)):
            return ({i:[t] for i,t in enumerate(data)} if strat=="individual"
                    else {0:list(data)})
        if isinstance(data, pd.DataFrame):
            joined = data[self.text_cols].astype(str).agg(" ".join, axis=1)
            if strat=="column":
                if col not in data.columns:
                    raise KeyError(f"cluster_col '{col}' missing")
                grp = data.assign(_j=joined).groupby(col,sort=False)["_j"].apply(list)
                return grp.to_dict()
            if strat=="individual":
                return {i:[t] for i,t in joined.items()}
            return {0:list(joined)}
        raise TypeError(f"Unsupported type {type(data)}")

    @staticmethod
    def _top_words(clusters: Dict[Any,List[str]], n: int=20) -> pd.DataFrame:
        def top(txts):
            toks = (w.lower() for t in txts for w in t.split()
                    if w.isalpha() and len(w)>2)
            return [w for w,_ in Counter(toks).most_common(n)]
        return pd.DataFrame({cid:pd.Series(top(txts)) for cid,txts in clusters.items()})

    def label_clusters(
        self,
        df: pd.DataFrame,
        kw_df: pd.DataFrame,
        *,
        provider: str,
        model: str,
        api_key: Optional[str],
        k: int,
        crit: Dict[str,Any],
        gpu: bool,
    ) -> Dict[int,str]:
        centres = self._centres(df, gpu)
        dist_kw = self._distinctive(df)
        out: Dict[int,str] = {}

        for cid in kw_df.columns:
            freq_kw = kw_df[cid].dropna().tolist()
            dist_kwl= dist_kw.get(int(cid),[])[:5]
            titles = sorted(
                df.loc[df["cluster"]==cid, self.text_cols]
                  .astype(str).agg(" ".join, axis=1),
                key=len
            )[:3]

            if provider=="ollama":
                cands = self._cand_ollama(freq_kw, dist_kwl, titles, crit, model, k)
            else:
                cands = self._cand_openai(freq_kw, dist_kwl, titles, crit, model, api_key, k)

            if cid in centres and cands:
                _, vec = centres[cid]
                out[int(cid)] = self._pick(cands, vec, gpu)

        return out

    def label_texts(
        self,
        data: Union[str, Iterable[str], Mapping[Any,str], pd.DataFrame],
        *,
        provider: str = "ollama",
        model_name: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        cluster_strategy: str = "single",
        cluster_col: str   = "cluster",
        top_n_words:  int   = 20,
        num_candidates: int = 12,
        use_gpu: bool       = False,
    ) -> Dict[int,str]:
        provider = provider.lower()
        if provider not in {"ollama","openai"}:
            raise ValueError("provider must be 'ollama' or 'openai'")
        if provider=="openai" and not openai_api_key:
            raise ValueError("openai_api_key required for OpenAI")

        clusters = self._to_clusters(data, strat=cluster_strategy, col=cluster_col)
        kw_df    = self._top_words(clusters, top_n_words)
        col0     = self.text_cols[0]
        rows = [
            {"cluster":cid, col0:txt, **{c:txt for c in self.text_cols[1:]}}
            for cid,txts in clusters.items() for txt in txts
        ]
        df   = pd.DataFrame(rows)

        crit  = self._crit()
        model = model_name or (
            "llama3.2:3b-instruct-fp16" if provider=="ollama" else "gpt-3.5-turbo"
        )
        return self.label_clusters(
            df, kw_df,
            provider=provider,
            model=model,
            api_key=openai_api_key,
            k=num_candidates,
            crit=crit,
            gpu=use_gpu,
        )

    # convenience wrappers
    def label_clusters_ollama(self, df, kw_df, **kw):
        return self.label_clusters(
            df, kw_df, provider="ollama",
            model=kw.get("model_name","llama3.2:3b-instruct-fp16"),
            api_key=None,
            k=kw.get("num_candidates",12),
            crit=self._crit(),
            gpu=kw.get("use_gpu",False),
        )

    def label_clusters_openai(self, df, kw_df, openai_api_key, **kw):
        return self.label_clusters(
            df, kw_df, provider="openai",
            model=kw.get("model_name","gpt-3.5-turbo"),
            api_key=openai_api_key,
            k=kw.get("num_candidates",12),
            crit=self._crit(),
            gpu=kw.get("use_gpu",False),
        )
