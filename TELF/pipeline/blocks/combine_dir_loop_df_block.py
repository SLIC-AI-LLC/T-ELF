from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Sequence
import pandas as pd

from .base_block import AnimalBlock
from .data_bundle import DataBundle, SAVE_DIR_BUNDLE_KEY, RESULTS_DEFAULT

class ConcatenateDFBlock(AnimalBlock):
    """
    Collect the latest *df* from every sub-bundle stored in ⟨results⟩
    and concatenate them row-wise, tagging each row with the
    great-grandparent folder name (e.g. "c5").
    """

    CANONICAL_NEEDS = (RESULTS_DEFAULT,)

    def __init__(
        self,
        *,
        needs: Sequence[str] = CANONICAL_NEEDS,
        provides: Sequence[str] = ("df",),
        source_key: str = "df",              # ← key to pull from each sub-bundle
        checkpoint_keys: Sequence[str] = ("df",),
        tag: str = "ConcatDF",
        **kw,
    ):
        self.source_key = source_key
        super().__init__(
            needs=needs,
            provides=provides,
            tag=tag,
            checkpoint_keys=checkpoint_keys,
            **kw,
        )

    def run(self, bundle: DataBundle) -> None:
        results_list = bundle[self.needs[0]]
        dfs: List[pd.DataFrame] = []

        for subdict in results_list:
            obj = subdict.get(self.source_key)
            if obj is None:
                raise KeyError(
                    f"Sub-bundle missing key {self.source_key!r}; "
                    f"available: {list(subdict)}"
                )

            # ── extract great-grandparent directory name ──
            try:
                p = Path(obj)
                # file → Orca → results → c5  ⇒  parents[3].name == "c5"
                source_dir = p.parent.parent.parent.name
            except Exception:
                source_dir = None

            df = self.load_path(obj)  # handles Path / CSV / NPZ / DF…
            if not isinstance(df, pd.DataFrame):
                raise TypeError(f"Expected DataFrame, got {type(df)}")

            # add the new column before concatenation
            df_copy = df.copy()
            df_copy["source_dir"] = source_dir

            dfs.append(df_copy)

        consolidated = pd.concat(dfs, ignore_index=True)

        # (optional) persist + checkpoint
        if SAVE_DIR_BUNDLE_KEY in bundle:
            out_dir = Path(bundle[SAVE_DIR_BUNDLE_KEY]) / self.tag
            out_dir.mkdir(parents=True, exist_ok=True)
            csv_path = out_dir / "consolidated_df.csv"
            consolidated.to_csv(csv_path, index=False)
            self.register_checkpoint(self.provides[0], csv_path)

        # save to bundle (namespaced + generic alias)
        bundle[f"{self.tag}.{self.provides[0]}"] = consolidated
