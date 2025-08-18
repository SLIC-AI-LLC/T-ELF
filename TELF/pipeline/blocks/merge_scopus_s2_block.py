# blocks/merge_scopus_s2_block.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

import pandas as pd
from ...helpers.frames import merge_scopus_s2

from .base_block import AnimalBlock
from .data_bundle import DataBundle


class MergeScopusS2Block(AnimalBlock):
    """
    Merge a *Scopus* DataFrame with a *Semantic Scholar (S2)* DataFrame.

    needs    : ('df', 'df')
               (override if your bundle uses different keys)
    provides : ('df',)
    tag      : 'MergeScopusS2'
    """

    # CANONICAL_NEEDS = ("Scopus.df", "S2.df")

    def __init__(
        self,
        *,
        needs: Sequence[str] =  ("Scopus.df", "S2.df"),
        provides: Sequence[str] = ("df",),
        conditional_needs: Sequence[Tuple[str, Any]] = (),
        tag: str = "MergeScopusS2",
        init_settings: Dict[str, Any] | None = None,
        call_settings: Dict[str, Any] | None = None,
        verbose: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            needs=needs,
            provides=provides,
            conditional_needs=conditional_needs,
            tag=tag,
            init_settings=self._merge({"verbose": True}, init_settings),
            call_settings=self._merge({}, call_settings),
            verbose=verbose,
            **kwargs,
        )

    # ------------------------------------------------------------------ #
    # work                                                                #
    # ------------------------------------------------------------------ #
    def run(self, bundle: DataBundle) -> None:
        src_left = bundle[self.needs[0]]
        src_right = bundle[self.needs[1]]

        left_df = self.load_path(src_left) if isinstance(src_left, (str, Path)) else src_left.copy()
        right_df = self.load_path(src_right) if isinstance(src_right, (str, Path)) else src_right.copy()

        merged_df = merge_scopus_s2(left_df, right_df, **self.call_settings)
        

        bundle[f"{self.tag}.{self.provides[0]}"] = merged_df
