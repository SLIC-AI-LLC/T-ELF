from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import warnings

import pandas as pd
import numpy as np

from .base_block import AnimalBlock
from .data_bundle import DataBundle, SAVE_DIR_BUNDLE_KEY
from ...applications.Cheetah import Cheetah
from ...pre_processing.Vulture.tokens_analysis.top_words import get_top_words


class TermAttributionBlock(AnimalBlock):
    """
    Runs Cheetah-based term representation + attribution on a DataFrame,
    including positive and negative dependents as additional terms.
    Processes both clean_title_abstract and raw title+abstract, and
    generates n-gram files for each, with counts in representation CSV.
    """
    CANONICAL_NEEDS = ("df", "terms")

    def __init__(
        self,
        *,
        needs: Tuple[str, ...] = CANONICAL_NEEDS,
        provides: Tuple[str, ...] = ("df", "term_representation_df"),
        tag: str = "Attribution",
        init_settings: Optional[Dict[str, Any]] = None,
        call_settings: Optional[Dict[str, Any]] = None,
    ):
        default_call = {
            "text_col": "clean_title_abstract",
            "cheetah_verbose": False,
            "ngram_top_n": 100
        }
        super().__init__(
            needs=needs,
            provides=provides,
            tag=tag,
            init_settings=init_settings or {},
            call_settings={**default_call, **(call_settings or {})},
        )

    def run(self, bundle: DataBundle) -> None:
        # 1) Load inputs
        df: pd.DataFrame = self.load_path(bundle[self.needs[0]])
        df = df.reset_index(drop=True)
        raw_terms = self.load_path(bundle[self.needs[1]])

        # 2) Parse main_terms and dependents
        main_terms: List[str] = []
        pos_terms: set[str] = set()
        neg_terms: set[str] = set()

        def extract_entry(val: Any) -> None:
            if isinstance(val, str):
                try:
                    parsed = ast.literal_eval(val)
                except Exception:
                    main_terms.append(val)
                    return
                if isinstance(parsed, dict) and len(parsed) == 1:
                    term, spec = next(iter(parsed.items()))
                else:
                    main_terms.append(val)
                    return
            elif isinstance(val, dict) and len(val) == 1:
                term, spec = next(iter(val.items()))
            else:
                raise TypeError(f"Unsupported term entry: {val!r}")

            main_terms.append(term)
            if isinstance(spec, dict):
                for p in spec.get('positives', []): pos_terms.add(p)
                for n in spec.get('negatives', []): neg_terms.add(n)
            elif isinstance(spec, list):
                for p in spec: pos_terms.add(p)

        if isinstance(raw_terms, pd.DataFrame):
            col0 = raw_terms.columns[0]
            for v in raw_terms[col0]: extract_entry(v)
        else:
            for entry in raw_terms: extract_entry(entry)

        # 3) Dedupe & order terms
        union_main_pos = set(main_terms) | pos_terms
        all_terms = (
            main_terms
            + [t for t in pos_terms if t not in main_terms]
            + [t for t in neg_terms if t not in union_main_pos]
        )
        seen = set()
        deduped_terms = [t for t in all_terms if not (t in seen or seen.add(t))]

        # 4) Validate clean text column
        text_col = self.call_settings['text_col']
        if text_col not in df.columns:
            raise KeyError(f"Missing text column {text_col!r}")

        # 5) Prepare Document Sets
        docs_clean = df[text_col].fillna("").astype(str).tolist()
        docs_raw = (
            df.get('title', pd.Series(dtype=str)).fillna("")
            + " " + df.get('abstract', pd.Series(dtype=str)).fillna("")
        ).astype(str).tolist()
        N_clean = len(docs_clean)
        N_raw = len(docs_raw)

        # 6) If nothing to process, build empty DataFrames
        if N_clean == 0 or not deduped_terms:
            term_representation_df = pd.DataFrame(columns=[
                'Term', 'TF_clean', 'DF_clean', 'TF_raw', 'DF_raw'
            ])
            attribution_df = pd.DataFrame(columns=list(df.columns) + ['term'])
        else:
            # 7) Compute TF counts via regex
            term_tfs_clean: Dict[str,int] = {}
            term_tfs_raw:   Dict[str,int] = {}
            total_tf_clean = 0
            total_tf_raw = 0

            for term in deduped_terms:
                pat = rf"\b{re.escape(term)}\b"
                tf_c = sum(len(re.findall(pat, d, flags=re.IGNORECASE)) for d in docs_clean)
                tf_r = sum(len(re.findall(pat, d, flags=re.IGNORECASE)) for d in docs_raw)
                term_tfs_clean[term] = tf_c
                term_tfs_raw[term]   = tf_r
                total_tf_clean += tf_c
                total_tf_raw   += tf_r

            # 8) Index only clean for DF via Cheetah
            cheetah_clean = Cheetah(verbose=self.call_settings['cheetah_verbose'])
            cheetah_clean.index(
                data=df.reset_index(drop=True),
                columns={'abstract': text_col},
                verbose=self.call_settings['cheetah_verbose']
            )

            repr_rows: List[Dict[str,Any]] = []
            attr_rows: List[Dict[str,Any]] = []

            for term in deduped_terms:
                tf_c = term_tfs_clean[term]
                tf_r = term_tfs_raw[term]
                pat = rf"\b{re.escape(term)}\b"

                # DF_clean via Cheetah.search
                df_clean = cheetah_clean.search(
                    query=[term], and_search=True,
                    in_title=False, in_abstract=True,
                    do_results_table=False
                )[0]
                df_c = len(df_clean)

                # DF_raw via regex count of documents
                df_r = sum(1 for d in docs_raw if re.search(pat, d, flags=re.IGNORECASE))

                repr_rows.append({
                    'Term': term,
                    'TF_clean': tf_c,
                    'DF_clean': df_c,
                    'TF_raw': tf_r,
                    'DF_raw': df_r,
                })

                # Attribution: one row per clean match
                for idx in df_clean.index:
                    base = df.iloc[idx].to_dict()
                    base['term'] = term
                    attr_rows.append(base)

            term_representation_df = pd.DataFrame.from_records(repr_rows)
            attribution_df = pd.DataFrame.from_records(attr_rows)

        # 9) Save CSV outputs
        base_dir = Path(bundle[SAVE_DIR_BUNDLE_KEY])
        out_dir = base_dir / self.tag
        out_dir.mkdir(parents=True, exist_ok=True)
        path_attr = out_dir / 'df_attribution.csv'
        path_repr = out_dir / 'attribution_representation.csv'
        attribution_df.to_csv(path_attr, index=False)
        term_representation_df.to_csv(path_repr, index=False)

        # 10) Generate n-gram files for both clean & raw
        top_n = self.call_settings['ngram_top_n']
        for mode, docs in [('clean', docs_clean), ('raw', docs_raw)]:
            for n in (1, 2, 3):
                ng = get_top_words(docs, top_n=top_n, n_gram=n, verbose=False)
                (out_dir / f'top_{n}-grams_{mode}.csv').write_text(
                    ng.iloc[:, :3].to_csv(index=False)
                )

        # 11) Register & bundle
        self.register_checkpoint(self.provides[0], path_attr)
        self.register_checkpoint(self.provides[1], path_repr)
        bundle[f"{self.tag}.{self.provides[0]}"] = attribution_df
        bundle[f"{self.tag}.{self.provides[1]}"] = term_representation_df 
