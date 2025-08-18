from pathlib import Path
from typing import Dict, Sequence, Any, Tuple
import pandas as pd
import os
import nest_asyncio
nest_asyncio.apply()

from .data_bundle import DataBundle, SAVE_DIR_BUNDLE_KEY

from ...pipeline.blocks.base_block import AnimalBlock
from ...helpers.frames import merge_scopus_s2, clean_duplicates
from  ...helpers.terms import form_scopus_search, form_s2_search  # adjust import to your project layout

from ...pre_processing.iPenguin.Scopus import Scopus
from ...pre_processing.iPenguin.SemanticScholar import SemanticScholar

class TermSearchBlock(AnimalBlock):
    CANONICAL_NEEDS = ("terms",)

    def __init__(
        self,
        needs=CANONICAL_NEEDS,
        provides=("df",),
        tag: str = "TermSearch",
        conditional_needs: Sequence[Tuple[str, Any]] = (),
        *,
        init_settings: Dict[str, Any] = None,
        call_settings: Dict[str, Any] = None,
        **kw,
    ):
        default_init = {
            'scopus':{
                'keys': None, #SCOPUS_KEYS, 
                'mode': 'fs',          # file system caching mode (default)
                'name': None, #SCOPUS_CACHE,  # where to cache the files
                'ignore': None, #scopus_ignore,
                'verbose': True  # run high level handler in debug (verbose > 10)
            },
            's2':{
                'key': None, #S2_KEY, 
                'mode': 'fs',          # file system caching mode (default)
                'name': None, #S2_CACHE,  # where to cache the files
                'ignore': None, #scopus_ignore,
                'verbose': True  # run high level handler in debug (verbose > 10)
            }
        }
        default_call = {}

        super().__init__(
            needs = needs,
            provides = provides,
            init_settings=self._merge(default_init, init_settings),
            call_settings=self._merge(default_call, call_settings),
            conditional_needs=conditional_needs,
            tag=tag,
            **kw,
        )

    def run(self, bundle:DataBundle) -> None:
        result_path = Path(bundle[SAVE_DIR_BUNDLE_KEY]) / self.tag
        result_path.mkdir(parents=True, exist_ok=True)

        terms = self.load_path(bundle[self.needs[0]])

        # === Step 1: Scopus search via TITLE-ABS
        scopus_outfile = result_path / "scopus.csv"
        if not scopus_outfile.exists():
            scopus = Scopus(**self.init_settings['scopus'])
            queries = [f"({form_scopus_search(term, code='TITLE-ABS')})" for term in terms]
            for q in queries:
                print(q)
            scopus_query = " OR ".join(queries)
            scopus_df, _ = scopus.search(scopus_query, n=0)
            scopus_df.to_csv(scopus_outfile, index=False)
        else:
            scopus_df = pd.read_csv(scopus_outfile)

        scopus_df = clean_duplicates(scopus_df)

        # === Step 2: S2 search
        s2_outfile = result_path / "s2.csv"
        if not s2_outfile.exists():
            s2 = SemanticScholar(**self.init_settings['s2'])
            s2_dfs = []
            for term in terms:
                q = form_s2_search(term)
                tmp_df, _ = s2.search(q, mode="bulk", n=0)
                s2_dfs.append(tmp_df)

            s2_df = pd.concat(s2_dfs, ignore_index=True)
            s2_df = s2_df.drop_duplicates(subset=["s2id"]).reset_index(drop=True)
            s2_df.to_csv(s2_outfile, index=False)
        else:
            s2_df = pd.read_csv(s2_outfile, lineterminator="\n")

        # === Step 3: Merge
        merged_df = merge_scopus_s2(scopus_df, s2_df)

        # === Save final
        final_outfile = result_path / "final_papers.csv"
        merged_df.to_csv(final_outfile, index=False)

        bundle[f"{self.tag}.{self.provides[0]}"] = merged_df
