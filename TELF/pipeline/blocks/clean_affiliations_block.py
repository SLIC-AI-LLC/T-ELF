# blocks/clean_affiliations_block.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

import pandas as pd
from ...helpers.filters import clean_affiliations

from .base_block import AnimalBlock
from .data_bundle import DataBundle


class CleanAffiliationsBlock(AnimalBlock):
    """
    Standardise and de-duplicate author affiliation strings.

    needs    : ('df',)
               (or any DataFrame key you choose)
    provides : ('df',)
    tag      : 'CleanAffiliations'
    """

    CANONICAL_NEEDS = ("df",)

    def __init__(
        self,
        *,
        needs: Sequence[str] = CANONICAL_NEEDS,
        provides: Sequence[str] = ("df",),
        conditional_needs: Sequence[Tuple[str, Any]] = (),
        tag: str = "CleanAffiliations",
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
        src = bundle[self.needs[0]]
        df = self.load_path(src) if isinstance(src, (str, Path)) else src.copy()

        fixed_df = clean_affiliations(df, **self.call_settings)

        bundle[f"{self.tag}.{self.provides[0]}"] = fixed_df
