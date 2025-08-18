# blocks/beaver_citation_tensor_block.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

import pandas as pd
from ...pre_processing.Beaver import Beaver
from .base_block import AnimalBlock
from .data_bundle import DataBundle, SAVE_DIR_BUNDLE_KEY


class BeaverCitationTensorBlock(AnimalBlock):
    """
    Build an **(authors × papers × references)** citation tensor with *Beaver*.

    The underlying call is ``beaver.citation_tensor`` which returns:

    * **X**          – 3-mode tensor
    * **authors**    – unique author identifiers (axis 0)
    * **paper_ids**  – paper / EID identifiers (axis 1)
    * **years**      – publication years (axis 2 or metadata)

    ─────────────────────────────────────────────────────────────
    needs        : ('df',)
    provides     : ('X', 'authors', 'paper_ids', 'years')
    tag          : 'BeaverCit'
    """
    CANONICAL_NEEDS = ("df",)

    # ------------------------------------------------------------------ #
    # constructor                                                         #
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        *,
        target_columns: Tuple[str, str, str, str] = (
            "author_ids",
            "eid",
            "references",
            "year",
        ),
        needs: Sequence[str] = CANONICAL_NEEDS,
        provides: Sequence[str] = ("X", "authors", "paper_ids", "years"),
        conditional_needs: Sequence[Tuple[str, Any]] = (),
        tag: str = "BeaverCit",
        init_settings: Dict[str, Any] | None = None,
        call_settings: Dict[str, Any] | None = None,
        verbose: bool = True,
        **kwargs: Any,
    ) -> None:

        # settings that go to Beaver(...)
        default_init = {
            "n_jobs": -1,
            "n_nodes": 1,
        }

        # settings for beaver.citation_tensor(...)
        default_call = {
            "target_columns": target_columns,
            "dimension_order": [0, 1, 2],
            "split_authors_with": ";",
            "split_references_with": ";",
            "verbose": False,
            "n_jobs": 1,
            "n_nodes": 1,
            # "joblib_backend": "multiprocessing",     # disabled (see note)
            # "save_path" injected at runtime if bundle['result_path'] exists
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
        # 1️⃣  Load / copy the DataFrame ------------------------------------
        src_df = bundle[self.needs[0]]
        df = self.load_path(src_df) if isinstance(src_df, (str, Path)) else src_df.copy()

        # 2️⃣  Ensure rows contain all required columns ---------------------
        for col in self.call_settings["target_columns"]:
            if col in df.columns:
                df = df.dropna(subset=[col])

        # 3️⃣  Prepare call specification -----------------------------------
        cb: Dict[str, Any] = dict(self.call_settings)

        # dynamic `save_path`
        if "save_path" not in cb and SAVE_DIR_BUNDLE_KEY in bundle:
            cb["save_path"] = Path(bundle[SAVE_DIR_BUNDLE_KEY]) / self.tag

        # 4️⃣  Execute Beaver -------------------------------------------------
        beaver = Beaver(**self.init_settings)
        X, authors, paper_ids, years = beaver.citation_tensor(dataset=df, **cb)

        # 5️⃣  Store results --------------------------------------------------
        ns = self.tag
        bundle[f"{ns}.{self.provides[0]}"] = X
        bundle[f"{ns}.{self.provides[1]}"] = authors
        bundle[f"{ns}.{self.provides[2]}"] = paper_ids
        bundle[f"{ns}.{self.provides[3]}"] = years
