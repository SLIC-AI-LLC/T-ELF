# blocks/beaver_cooccurrence_block.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

import pandas as pd
from ...pre_processing.Beaver import Beaver
from .base_block import AnimalBlock
from .data_bundle import DataBundle, SAVE_DIR_BUNDLE_KEY


class BeaverCooccurrenceBlock(AnimalBlock):
    """
    Build a **co-occurrence** (and SPPMI) matrix with *Beaver*.

    ─────────────────────────────────────────────────────────────
    needs        : ('df', 'vocabulary')
    provides     : ('cooccurrence', 'sppmi')
    tag          : 'BeaverCooc'
    """
    CANONICAL_NEEDS = ("df", "vocabulary")

    # ------------------------------------------------------------------ #
    # constructor                                                         #
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        *,
        target_column: str = "clean_abstract",
        needs: Sequence[str] = CANONICAL_NEEDS,
        provides: Sequence[str] = ("cooccurrence", "sppmi"),
        conditional_needs: Sequence[Tuple[str, Any]] = (),
        tag: str = "BeaverCooc",
        init_settings: Dict[str, Any] | None = None,
        call_settings: Dict[str, Any] | None = None,
        verbose: bool = True,
        **kwargs: Any,
    ) -> None:

        # settings sent to Beaver(...)
        default_init = {
            "n_jobs": -1,
            "n_nodes": 1,
        }

        # settings sent to beaver.cooccurrence_matrix(...)
        default_call = {
            "target_column": target_column,
            "cooccurrence_settings": {
                "n_jobs": 2,
                "window_size": 100,
                # 'vocabulary' injected at runtime
            },
            "sppmi_settings": {},
            # "save_path" will be added automatically if bundle[SAVE_DIR_BUNDLE_KEY] exists
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
        # 1️⃣  Fetch dataframe ----------------------------------------------
        src_df = bundle[self.needs[0]]
        if isinstance(src_df, (str, Path)):
            df = self.load_path(src_df)
        else:
            df = src_df.copy()

        # 2️⃣  Fetch vocabulary --------------------------------------------
        vocabulary = bundle[self.needs[1]]

        # 3️⃣  Drop rows without the target column --------------------------
        target_col = self.call_settings["target_column"]
        df = df.dropna(subset=[target_col])

        # 4️⃣  Prepare Beaver & call spec -----------------------------------
        beaver = Beaver(**self.init_settings)

        cb: Dict[str, Any] = dict(self.call_settings)

        # inject vocabulary into nested dict
        cb.setdefault("cooccurrence_settings", {})
        cb["cooccurrence_settings"]["vocabulary"] = vocabulary

        # dynamic save path
        if "save_path" not in cb and SAVE_DIR_BUNDLE_KEY in bundle:
            cb["save_path"] = Path(bundle[SAVE_DIR_BUNDLE_KEY]) / self.tag

        # 5️⃣  Build matrices ------------------------------------------------
        cooc_mat, sppmi_mat = beaver.cooccurrence_matrix(dataset=df, **cb)

        # 6️⃣  Store outputs under namespace --------------------------------
        ns = self.tag
        bundle[f"{ns}.{self.provides[0]}"] = cooc_mat   # 'cooccurrence'
        bundle[f"{ns}.{self.provides[1]}"] = sppmi_mat  # 'sppmi'
