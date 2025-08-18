# blocks/beaver_something_words_time_block.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

import pandas as pd
from ...pre_processing.Beaver import Beaver
from .base_block import AnimalBlock
from .data_bundle import DataBundle, SAVE_DIR_BUNDLE_KEY


class BeaverSomethingWordsTimeBlock(AnimalBlock):
    """
    Build a **(something × words × time)** tensor with *Beaver*.

    The underlying call is ``beaver.something_words_time`` which returns:

    * **X**            – 3-mode tensor (authors × words × time)
    * **keys**         – sequence of “something” IDs (e.g.\ author_ids)
    * **vocabulary**   – updated vocabulary list / dict
    * **time_index**   – sequence of temporal labels (e.g.\ years)

    ─────────────────────────────────────────────────────────────
    needs        : ('df', 'vocabulary')
    provides     : ('X', 'keys', 'vocabulary', 'time_index')
    tag          : 'BeaverSWT'
    """
    CANONICAL_NEEDS = ("df", "vocabulary")

    # ------------------------------------------------------------------ #
    # constructor                                                         #
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        *,
        target_columns: Tuple[str, str, str] = ("author_ids", "clean_abstract", "year"),
        needs: Sequence[str] = CANONICAL_NEEDS,
        provides: Sequence[str] = ("X", "keys", "vocabulary", "time_index"),
        conditional_needs: Sequence[Tuple[str, Any]] = (),
        tag: str = "BeaverSWT",
        init_settings: Dict[str, Any] | None = None,
        call_settings: Dict[str, Any] | None = None,
        verbose: bool = True,
        **kwargs: Any,
    ) -> None:

        # Settings passed to Beaver(...)
        default_init = {
            "n_jobs": -1,
            "n_nodes": 1,
        }

        # Settings passed to beaver.something_words_time(...)
        default_call = {
            "target_columns": target_columns,
            # vocabulary injected at runtime
            "tfidf_transformer": True,
            "unfold_at": 1,
            "verbose": False,
            # "save_path" set automatically if bundle["result_path"] exists
        }

        super().__init__(
            needs=needs,
            provides=provides,
            conditional_needs=conditional_needs,
            tag=tag,
            init_settings=self._merge(default_init, init_settings),
            call_settings=self._merge(default_call, call_settings),
            verbose=verbose,
            **kwargs,
        )

    # ------------------------------------------------------------------ #
    # work                                                                #
    # ------------------------------------------------------------------ #
    def run(self, bundle: DataBundle) -> None:
        # 1️⃣  Load DataFrame -------------------------------------------------
        src_df = bundle[self.needs[0]]
        df = self.load_path(src_df) if isinstance(src_df, (str, Path)) else src_df.copy()

        # 2️⃣  Obtain vocabulary --------------------------------------------
        vocab = bundle[self.needs[1]]

        # 3️⃣  Drop rows missing any target column ---------------------------
        for col in self.call_settings["target_columns"]:
            if col in df.columns:
                df = df.dropna(subset=[col])

        # 4️⃣  Build call-time settings --------------------------------------
        cb: Dict[str, Any] = dict(self.call_settings)
        cb["vocabulary"] = vocab

        # save_path → result_path/BeaverSWT if none supplied
        if "save_path" not in cb and SAVE_DIR_BUNDLE_KEY in bundle:
            cb["save_path"] = Path(bundle[SAVE_DIR_BUNDLE_KEY]) / self.tag

        # 5️⃣  Execute Beaver -------------------------------------------------
        beaver = Beaver(**self.init_settings)
        X, keys, vocab_new, time_axis = beaver.something_words_time(dataset=df, **cb)

        # 6️⃣  Store outputs --------------------------------------------------
        ns = self.tag
        bundle[f"{ns}.{self.provides[0]}"] = X
        bundle[f"{ns}.{self.provides[1]}"] = keys
        bundle[f"{ns}.{self.provides[2]}"] = vocab_new
        bundle[f"{ns}.{self.provides[3]}"] = time_axis
