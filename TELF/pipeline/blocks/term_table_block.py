from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Callable
import pandas as pd
import numpy as np

from .base_block import AnimalBlock
from .data_bundle import DataBundle, SAVE_DIR_BUNDLE_KEY
from ...applications import Ocelot  # your Ocelot class


class TermTableBlock(AnimalBlock):
    """
    Build a term-presence matrix aligned with Ocelot:

      - Rows: documents
      - Columns: eid, doi, s2id + one column per search term from terms.md
      - Cell value: the term itself if matched in text; otherwise "_"

    Matching mirrors Ocelot exactly:
      * phrases parsed via Ocelot.parse_rules_markdown
      * (prefer) patterns via Ocelot.phrase_to_pattern_str when direct compile seems inert
      * one grouped regex via Ocelot._compile_group_pattern
      * scanning via Ocelot._scan (finditer on the compiled pattern)
      * text normalization via Ocelot’s *bound* preprocessor when available
      * text column is rebuilt the same way as OcelotFilterBlock (fallback/title+abstract)

    This eliminates discrepancies where Ocelot finds hits but the term table shows none.
    """

    CANONICAL_NEEDS = ("df", "term_path")

    def __init__(
        self,
        *,
        needs=CANONICAL_NEEDS,
        provides=("terms_list", "terms_table"),
        id_field: str = "eid",
        text_field: str = "text",
        doi_field: str = "doi",
        s2id_field: str = "s2id",
        # NEW: rebuild text like OcelotFilterBlock
        text_columns: Optional[Dict[str, str]] = None,
        tag: str = "TermTable",
        init_settings: Optional[Dict[str, Any]] = None,
        **kw,
    ):
        self.id_field = id_field
        self.text_field = text_field
        self.doi_field = doi_field
        self.s2id_field = s2id_field

        self.text_columns = text_columns or {
            "title": "title",
            "abstract": "abstract",
            "fallback_text": "title_abstract",
        }

        defaults = {"verbose": True}
        super().__init__(
            needs=needs,
            provides=provides,
            tag=tag,
            init_settings=self._merge(defaults, init_settings),
            call_settings={},
            **kw,
        )

    # ---------------- helpers ----------------

    @staticmethod
    def _dedupe_preserve_order(seq: List[str]) -> List[str]:
        seen = set()
        out: List[str] = []
        for s in seq:
            k = s.casefold()
            if k not in seen:
                seen.add(k)
                out.append(s)
        return out

    def _collect_all_phrases(self, md_text: str) -> List[str]:
        """All phrases from Ocelot markdown (mains, pos, neg, global pos/neg), deduped (casefold)."""
        rules, gpos, gneg = Ocelot.parse_rules_markdown(md_text, default_positives_mode="any")
        phrases: List[str] = []
        for r in rules:
            phrases.extend(r.main_phrases)
            phrases.extend(r.positive_phrases)
            phrases.extend(r.negative_phrases)
        phrases.extend(gpos)
        phrases.extend(gneg)
        phrases = [p.strip() for p in phrases if p and p.strip()]
        return self._dedupe_preserve_order(phrases)

    def _rebuild_text_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        ALWAYS rebuild df[self.text_field] to match OcelotFilterBlock:
        Priority: fallback_text -> (title + abstract) -> "".
        Any pre-existing text column is ignored and dropped.
        """
        tf = self.text_field
        title = self.text_columns.get("title", "title")
        abstract = self.text_columns.get("abstract", "abstract")
        fallback = self.text_columns.get("fallback_text", "title_abstract")

        if tf in df.columns:
            df = df.drop(columns=[tf])
            print(f"[{self.tag}] dropped existing text field '{tf}' (forced rebuild)")

        if fallback in df.columns:
            df[tf] = df[fallback].fillna("").astype(str)
            print(f"[{self.tag}] rebuilt '{tf}' from fallback '{fallback}'")
            return df

        has_title = title in df.columns
        has_abstract = abstract in df.columns
        if has_title or has_abstract:
            t_series = df.get(title, pd.Series("", index=df.index)).fillna("").astype(str)
            a_series = df.get(abstract, pd.Series("", index=df.index)).fillna("").astype(str)
            df[tf] = (t_series + " " + a_series).str.strip()
            print(f"[{self.tag}] rebuilt '{tf}' by merging '{title}' + '{abstract}'")
            return df

        df[tf] = ""
        print(
            f"[{self.tag}] WARNING: no '{fallback}', '{title}', or '{abstract}' columns; "
            f"rebuilt '{tf}' as empty strings"
        )
        return df

    def _get_prep_fn(self, oc_instance=None) -> Tuple[Callable[[str], str], str]:
        """
        Prefer a *bound* Ocelot instance method (exactly what OcelotFilter uses).
        Fall back to class-level function if present; else identity.
        """
        names = ("_prep_text", "_preprocess", "preprocess_text", "normalize_text", "prepare_text", "_normalize")
        if oc_instance is not None:
            for name in names:
                fn = getattr(oc_instance, name, None)
                if callable(fn):
                    return fn, f"oc.{name}"
        for name in names:
            fn = getattr(Ocelot, name, None)
            if callable(fn):
                return fn, f"Ocelot.{name}"
        return (lambda x: x), "identity"

    # ---------------- run ----------------

    def run(self, bundle: DataBundle) -> None:
        # 1) Load inputs
        df: pd.DataFrame = self.load_path(bundle[self.needs[0]])
        terms_path: Path = bundle[self.needs[1]]
        md_text = Path(terms_path).read_text(encoding="utf-8")

        print(f"\n[{self.tag}] ====================================================")
        print(f"[{self.tag}] df.shape          = {df.shape}")
        print(f"[{self.tag}] terms_md          = {terms_path}")

        # 2) Ensure required columns
        if self.id_field not in df.columns:
            raise ValueError(f"DataFrame must contain '{self.id_field}' column.")

        # Rebuild text like OcelotFilterBlock (so both scan the same strings)
        df = self._rebuild_text_column(df)

        if self.doi_field not in df.columns:
            df[self.doi_field] = ""
            print(f"[{self.tag}] NOTE: '{self.doi_field}' not found; created empty column.")
        if self.s2id_field not in df.columns:
            df[self.s2id_field] = ""
            print(f"[{self.tag}] NOTE: '{self.s2id_field}' not found; created empty column.")

        nonempty = (df[self.text_field].str.len() > 0).sum()
        print(f"[{self.tag}] non-empty texts   = {nonempty} / {len(df)}")

        # 3) Collect phrases
        phrases = self._collect_all_phrases(md_text)
        print(f"[{self.tag}] total phrases     = {len(phrases)}")
        print(f"[{self.tag}] sample phrases    = {phrases[:10]}")

        # Produce id-only table if no phrases
        if not phrases:
            out = df[[self.id_field, self.doi_field, self.s2id_field]].copy()
            self._save_and_register(bundle, out, phrases)
            return

        # 4) Build an Ocelot engine and get its *bound* normalizer
        oc = Ocelot.from_markdown(
            md_text,
            positives_mode="any",
            global_positives_mode="any",
        )
        prep_fn, prep_name = self._get_prep_fn(oc)
        print(f"[{self.tag}] using prep_fn      = {prep_name}")

        # Prepare a small normalized probe to sanity-check the regex
        probe_src = " ".join(df[self.text_field].fillna("").astype(str).head(5).tolist())
        probe_norm = prep_fn(probe_src)

        # 5) Compile ONE grouped regex (resilient path)
        compiled_from = "phrases"
        any_rx, gmap = Ocelot._compile_group_pattern(phrases)
        if any_rx is None or (probe_norm and not any_rx.search(probe_norm)):
            try:
                patterns = [Ocelot.phrase_to_pattern_str(p) for p in phrases]
                any_rx2, gmap2 = Ocelot._compile_group_pattern(patterns)
                if any_rx2 is not None:
                    any_rx, gmap = any_rx2, gmap2
                    compiled_from = "patterns"
            except Exception as e:
                # keep original compile result; will fall through if still unusable
                print(f"[{self.tag}] fallback compile failed with: {e!r}")

        if any_rx is None:
            # no valid regex chunks → id-only
            print(f"[{self.tag}] WARNING: compiled regex is None; emitting id-only table.")
            out = df[[self.id_field, self.doi_field, self.s2id_field]].copy()
            self._save_and_register(bundle, out, phrases)
            return

        has_probe = bool(any_rx.search(probe_norm)) if probe_norm else False
        print(f"[{self.tag}] compiled_from     = {compiled_from}")
        print(f"[{self.tag}] probe has match?  = {has_probe}")

        # 6) Build wide matrix: start with ids, fill each term column with "_"
        out_cols = [self.id_field, self.doi_field, self.s2id_field]
        out = df[out_cols].copy()

        # Initialize all term columns with "_"
        for term in phrases:
            out[term] = "_"

        # 7) Scan each row with Ocelot._scan (label-based assignment to avoid misalignment)
        text_series = df[self.text_field].fillna("").astype(str)
        rows_with_hits = 0

        for row_idx, txt in text_series.items():
            txt_norm = prep_fn(txt)
            hit_idxs = Ocelot._scan(any_rx, gmap, txt_norm)  # -> Set[int] of phrase indices
            if hit_idxs:
                rows_with_hits += 1
                for j in hit_idxs:
                    term = phrases[j]
                    out.loc[row_idx, term] = term

        print(f"[{self.tag}] rows with ≥1 match = {rows_with_hits} / {len(out)}")

        # 8) Save & register
        self._save_and_register(bundle, out, phrases)

    # ---------------- IO helpers ----------------

    def _save_and_register(self, bundle: DataBundle, out: pd.DataFrame, phrases: List[str]) -> None:
        root = Path(bundle.get(SAVE_DIR_BUNDLE_KEY, "."))
        outdir = root / self.tag
        outdir.mkdir(parents=True, exist_ok=True)

        df_path = outdir / f"{self.tag}.csv"
        terms_table_path = outdir / "terms_table.csv"

        out.to_csv(df_path, index=False, encoding="utf-8-sig")
        pd.DataFrame({"term": phrases}).to_csv(terms_table_path, index=False, encoding="utf-8-sig")

        self.register_checkpoint(self.provides[0], df_path)
        self.register_checkpoint(self.provides[1], terms_table_path)
        bundle[f"{self.tag}.{self.provides[0]}"] = out
        bundle[f"{self.tag}.{self.provides[1]}"] = pd.DataFrame({"term": phrases})

        print(f"[{self.tag}] saved matrix      → {df_path}")
        print(f"[{self.tag}] saved terms_table → {terms_table_path}")
