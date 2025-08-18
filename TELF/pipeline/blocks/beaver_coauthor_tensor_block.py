# blocks/beaver_coauthor_tensor_block.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

import pandas as pd
from ...pre_processing.Beaver import Beaver
from .base_block import AnimalBlock
from .data_bundle import DataBundle, SAVE_DIR_BUNDLE_KEY


class BeaverCoauthorTensorBlock(AnimalBlock):
    """
    Build an **(authors × authors × time)** co-author tensor with *Beaver*.

    The underlying call is ``beaver.coauthor_tensor`` which returns:

    * **X**        – 3-mode tensor (author i × author j × year)
    * **authors**  – unique author identifiers (shared by modes 0 & 1)
    * **years**    – temporal index (e.g. publication years)

    ─────────────────────────────────────────────────────────────
    needs        : ('df',)
    provides     : ('X', 'authors', 'years')
    tag          : 'BeaverCoAuth'
    """
    CANONICAL_NEEDS = ("df",)

    # ------------------------------------------------------------------ #
    # constructor                                                         #
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        *,
        target_columns: Tuple[str, str] = ("author_ids", "year"),
        needs: Sequence[str] = CANONICAL_NEEDS,
        provides: Sequence[str] = ("X", "authors", "years"),
        conditional_needs: Sequence[Tuple[str, Any]] = (),
        tag: str = "BeaverCoAuth",
        init_settings: Dict[str, Any] | None = None,
        call_settings: Dict[str, Any] | None = None,
        verbose: bool = True,
        **kwargs: Any,
    ) -> None:

        # Parameters passed to Beaver(...)
        default_init = {
            "n_jobs": -1,
            "n_nodes": 1,
        }

        # Parameters forwarded to beaver.coauthor_tensor(...)
        default_call = {
            "target_columns": target_columns,
            "split_authors_with": ";",
            "verbose": False,
            "n_jobs": -1,
            "n_nodes": 1,
            # "save_path" injected dynamically if bundle['result_path'] exists
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
        # 1️⃣  Get the dataframe --------------------------------------------
        src_df = bundle[self.needs[0]]
        df = self.load_path(src_df) if isinstance(src_df, (str, Path)) else src_df.copy()

        # 2️⃣  Drop rows missing required columns ----------------------------
        for col in self.call_settings["target_columns"]:
            if col in df.columns:
                df = df.dropna(subset=[col])

        # 3️⃣  Build call dictionary -----------------------------------------
        cb: Dict[str, Any] = dict(self.call_settings)

        # dynamic save_path
        if "save_path" not in cb and SAVE_DIR_BUNDLE_KEY in bundle:
            cb["save_path"] = Path(bundle[SAVE_DIR_BUNDLE_KEY]) / self.tag

        # 4️⃣  Execute Beaver -------------------------------------------------
        beaver = Beaver(**self.init_settings)
        X, authors, years = beaver.coauthor_tensor(dataset=df, **cb)

        # 5️⃣  Store outputs --------------------------------------------------
        ns = self.tag
        bundle[f"{ns}.{self.provides[0]}"] = X
        bundle[f"{ns}.{self.provides[1]}"] = authors
        bundle[f"{ns}.{self.provides[2]}"] = years
