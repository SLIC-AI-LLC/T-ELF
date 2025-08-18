# blocks/filter_by_count_block.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Sequence, Tuple, Optional

import pandas as pd
from .base_block import AnimalBlock
from .data_bundle import DataBundle


# ------------------------------------------------------------------ #
# helper                                                             #
# ------------------------------------------------------------------ #
def filter_by_count(
    df: pd.DataFrame,
    col: str = "author_ids",
    min_count: Optional[int] = None,
    max_count: Optional[int] = 30,
    sep: Optional[str] = None,
) -> pd.DataFrame:
    """See full docstring in original question."""
    sep = sep or ";"

    def _count(cell: Any) -> int:
        if pd.isna(cell):
            return 0
        if isinstance(cell, (list, tuple, set)):
            return len(cell)
        if isinstance(cell, str):
            return len([p for p in cell.split(sep) if p])
        return 1  # anything else

    counts = df[col].apply(_count)
    mask = pd.Series(True, index=df.index)
    if min_count is not None:
        mask &= counts >= min_count
    if max_count is not None:
        mask &= counts <= max_count
    return df.loc[mask].copy()


# ------------------------------------------------------------------ #
# block                                                              #
# ------------------------------------------------------------------ #
class FilterByCountBlock(AnimalBlock):
    """
    Filter a DataFrame by counting items in a column.

    needs        : ('df',)
    provides     : ('df',)
    tag          : 'FilterByCount'
    conditional  : none (for now)
    """
    CANONICAL_NEEDS = ('df', )
    
    def __init__(
        self,
        *,
        col: str = "author_ids",
        needs: Sequence[str] = CANONICAL_NEEDS,
        provides: Sequence[str] = ("df",),
        conditional_needs: Sequence[Tuple[str, Any]] = (),  # empty
        tag: str = "FilterByCount",
        init_settings: Dict[str, Any] | None = None,
        call_settings: Dict[str, Any] | None = None,
        verbose: bool = True,
        **kwargs: Any,
    ) -> None:
        self.col = col

        default_init = {"verbose": True}
        default_call = {
            "col": self.col,
            "min_count": None,
            "max_count": 30,
            "sep": None,
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
        # 1️⃣  pull the DataFrame (generic or namespaced)
        src = bundle[self.needs[0]]
        df = self.load_path(src) if isinstance(src, (str, Path)) else src.copy()

        # 2️⃣  filter
        filtered_df = filter_by_count(df, **self.call_settings)

        # 3️⃣  store under this block’s namespace
        bundle[f"{self.tag}.{self.provides[0]}"] = filtered_df
