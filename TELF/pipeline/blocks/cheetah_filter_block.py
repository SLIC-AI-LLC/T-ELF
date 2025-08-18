from pathlib import Path
from typing import Dict, Any, Optional
import json, warnings, pprint
import pandas as pd

from .base_block import AnimalBlock
from .data_bundle import DataBundle, SAVE_DIR_BUNDLE_KEY
from ...applications import Cheetah


class CheetahFilterBlock(AnimalBlock):
    """
    Hop-aware Cheetah filter with ultra-verbose logging.
    """

    CANONICAL_NEEDS = ("df", "query")

    # ----------------------------------------------------- INIT
    def __init__(
        self,
        *,
        needs=CANONICAL_NEEDS,
        provides=("df", "cheetah_table"),
        cheetah_columns: Optional[Dict[str, str]] = None,
        tag: str = "CheetahFilter",
        init_settings: Optional[Dict[str, Any]] = None,
        call_settings: Optional[Dict[str, Any]] = None,
        **kw,
    ):
        default_columns = {
            "title":      "title_abstract",
            "abstract":   "clean_title_abstract",
            "year":       "year",
            "author_ids": "author_ids",
        }
        self.cheetah_columns = cheetah_columns or default_columns

        default_init = {"verbose": False, "use_hops": False}
        default_call = {
            "in_title":           True,
            "in_abstract":        True,
            "and_search":         False,
            "ngram_window_size":  None,
            "ngram_ordered":      False,
            "do_results_table":   True,
        }

        super().__init__(
            needs=needs,
            provides=provides,
            tag=tag,
            init_settings=self._merge(default_init, init_settings),
            call_settings=self._merge(default_call, call_settings),
            **kw,
        )

    # ------------------------------------------------------ RUN
    def run(self, bundle: DataBundle) -> None:

        # 1 ─ load inputs

        df    = self.load_path(bundle[self.needs[0]])
        df.info()
        query = self.load_path(bundle[self.needs[1]])

        print(f"\n[{self.tag}] ====================================================")
        print(f"[{self.tag}] df.shape          = {df.shape}")
        if isinstance(query, (list, tuple)):
            print(f"[{self.tag}] query len        = {len(query)}")
            for i, q in enumerate(query, 1):
                print(f"[{self.tag}]   • q[{i:02d}] = {q!r}")
        else:
            print(f"[{self.tag}] query            = {query!r}")

        if "eid" not in df.columns:
            raise ValueError("DataFrame must contain an 'eid' column.")

        cheetah_kwargs = dict(self.call_settings, query=query)

        # 2 ─ hop split
        use_hops = self.init_settings["use_hops"]
        if use_hops and "type" in df.columns:
            max_t   = df.type.max()
            df_prev = df[df.type < max_t]
            df_curr = df[df.type == max_t].reset_index(drop=True)
            print(f"[{self.tag}] hop split        → prev {df_prev.shape}, curr {df_curr.shape}")
        else:
            df_prev = df.iloc[0:0]
            df_curr = df.reset_index(drop=True)
            print(f"[{self.tag}] no hop split     → curr {df_curr.shape}")

        # 3 ─ year numeric
        year_col = self.cheetah_columns["year"]
        df_curr[year_col] = pd.to_numeric(df_curr[year_col], errors="coerce")
        print(f"[{self.tag}] '{year_col}' numeric (NaNs={df_curr[year_col].isna().sum()})")

        # 4 ─ text columns
        title_col, abstract_col = self.cheetah_columns["title"], self.cheetah_columns["abstract"]
        if title_col not in df_curr.columns:
            df_curr[title_col] = (
                df_curr.get("title", pd.Series("", index=df_curr.index)).fillna("") + " " +
                df_curr.get("abstract", pd.Series("", index=df_curr.index)).fillna("")
            )
            print(f"[{self.tag}] built '{title_col}' by merging title+abstract")
        for col in (title_col, abstract_col):
            df_curr[col] = df_curr[col].fillna("").astype(str)
        print(f"[{self.tag}] text columns ready")

        # 5 ─ output dir & index
        root = Path(bundle.get(SAVE_DIR_BUNDLE_KEY, "."))
        out  = root / self.tag
        out.mkdir(parents=True, exist_ok=True)
        idx_file = out / f"{self.tag}_index.p"
        print(f"[{self.tag}] output dir        = {out}")

        cheetah = Cheetah(verbose=self.init_settings["verbose"])
        cheetah.index(
            df_curr,
            columns={k: v for k, v in self.cheetah_columns.items() if k in Cheetah.COLUMNS},
            index_file=str(idx_file),
            reindex=True,
        )
        print(f"[{self.tag}] indexed rows      = {len(df_curr)} → {idx_file.name}")

        # 5b ─ token hit map
        def _token_hits(ch_obj, qlist):
            print(f"[{self.tag}] ─ token hit map ─────────────────────────────")
            items = qlist if isinstance(qlist, (list, tuple)) else [qlist]
            for qi, q in enumerate(items, 1):
                key = next(iter(q)) if isinstance(q, dict) else q
                for tok in key.split():
                    t = tok.lower()
                    hit_t = len(ch_obj.title_index.get(t, []))
                    hit_a = len(ch_obj.abstract_index.get(t, []))
                    print(f"[{self.tag}]   q{qi:02d}:{t!r:<20} "
                          f"title={hit_t:4d}  abstract={hit_a:4d}")
            print(f"[{self.tag}] ─────────────────────────────────────────────")
        _token_hits(cheetah, query)

        # 6 ─ window size
        num_tok = len(str(query).split())
        if (ws := cheetah_kwargs.get("ngram_window_size")) is None or ws > num_tok:
            cheetah_kwargs["ngram_window_size"] = num_tok
        print(f"[{self.tag}] cheetah_kwargs:")
        pprint.pp(cheetah_kwargs, compact=True, width=100)

        # 7 ─ search
        try:
            cheetah_df, cheetah_table = cheetah.search(**cheetah_kwargs)
        except AssertionError as e:
            warnings.warn(f"{self.tag}: n-gram assertion – {e}")
            cheetah_df    = df_curr.iloc[0:0].copy()
            cheetah_table = pd.DataFrame(columns=["query"])
        print(f"[{self.tag}] search rows       = {len(cheetah_df)}")

        if cheetah_table is not None and "filter_type" in cheetah_table.columns:
            for _, r in cheetah_table.iterrows():
                if r["filter_type"] == "query":
                    print(f"[{self.tag}]   query='{r['filter_value']}' hits={r['num_papers']}")

        # 7b ─ explain table enrich
        if cheetah_table is not None and "included_ids" in cheetah_table.columns:
            ids_series = cheetah_table["included_ids"].fillna("").astype(str)
            cheetah_table["included_pos"] = ids_series.str.split(";").map(
                lambda L: [int(x) for x in L if x.isdigit()]
            )
            print(f"[{self.tag}] cheetah_table    rows={len(cheetah_table)}")

        # 8 ─ merge & dedupe
        merged = (
            pd.concat([df_prev, cheetah_df], ignore_index=True)
            .drop_duplicates(subset=["eid"], keep="last")
            .reset_index(drop=True)
        )
        print(f"[{self.tag}] merge result      = {merged.shape}")

        # 9 ─ refill columns
        orig = df.drop_duplicates(subset=["eid"], keep="first").set_index("eid")
        for col in orig.columns:
            merged[col] = merged[col].fillna(merged["eid"].map(orig[col].to_dict()))
        print(f"[{self.tag}] columns refilled")

        # 10 ─ save artefacts
        merged_path = out / f"{self.tag}.csv"
        merged.to_csv(merged_path, index=False, encoding="utf-8-sig")
        self.register_checkpoint(self.provides[0], merged_path)
        bundle[f"{self.tag}.{self.provides[0]}"] = merged
        print(f"[{self.tag}] saved df          → {merged_path}")

        if cheetah_table is not None:
            table_path = out / "cheetah_table.csv"
            cheetah_table.to_csv(table_path, index=False, encoding="utf-8-sig")
            self.register_checkpoint(self.provides[1], table_path)
            bundle[f"{self.tag}.{self.provides[1]}"] = cheetah_table
            print(f"[{self.tag}] saved table       → {table_path}")
        else:
            bundle[f"{self.tag}.{self.provides[1]}"] = None

        # save query as JSON
        query_path = out / "query.json"
        with open(query_path, "w", encoding="utf-8") as f:
            json.dump(query, f, ensure_ascii=False, indent=2)
        print(f"[{self.tag}] saved query       → {query_path}")
