# blocks/beaver_vocab_block.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Sequence

from .base_block import AnimalBlock
from .data_bundle import DataBundle, SAVE_DIR_BUNDLE_KEY
from ...pre_processing.Beaver import Beaver


class BeaverVocabBlock(AnimalBlock):
    """
    Build a TF-IDF vocabulary from a DataFrame column.

    ─────────────────────────────────────────────────────────────
    needs     : ('df',)              – the *latest* df in the bundle
    provides  : ('vocabulary',)      – a list/dict of tokens
    tag       : 'BeaverVocab'        – namespace for its outputs
    """
    CANONICAL_NEEDS = ("df",)

    # NOTE: we have *no* conditional needs for this block.  The argument is
    # left in place so future versions can add them without changing callers.
    def __init__(
        self,
        *,
        col: str = "clean_title_abstract",
        needs: Sequence[str] = CANONICAL_NEEDS,
        provides: Sequence[str] = ("vocabulary",),
        conditional_needs: Sequence[tuple[str, Any]] = (),
        tag: str = "BeaverVocab",
        init_settings: Dict[str, Any] | None = None,
        call_settings: Dict[str, Any] | None = None,
        **kw,
    ) -> None:

        self.col = col
        default_init = {
            "n_jobs": -1,
            "n_nodes": 1,
        }

        default_call = {
            "target_column": self.col,
            "max_df": 0.8,
            "min_df": 10,
            "max_features": 10_000,
        }

        super().__init__(
            needs=needs,
            provides=provides,
            conditional_needs=conditional_needs,  # currently empty
            tag=tag,
            init_settings=self._merge(default_init, init_settings),
            call_settings=self._merge(default_call, call_settings),
            **kw,
        )

    # ------------------------------------------------------------------ #
    # main work                                                           #
    # ------------------------------------------------------------------ #
    def run(self, bundle: DataBundle) -> None:
        # fetch the DataFrame (generic or namespaced – resolved by DataBundle)
        src_df = bundle[self.needs[0]]
        if isinstance(src_df, (str, Path)):
            df = self.load_path(src_df)
        else:
            df = src_df.copy()

        # clean up
        target = self.call_settings["target_column"]
        df = df.dropna(subset=[target])

        # generate vocabulary
        beaver = Beaver(**self.init_settings)

        # choose a save-to path if caller did not override it
        call_cfg = dict(self.call_settings)
        if "save_path" not in call_cfg and SAVE_DIR_BUNDLE_KEY in bundle:
            call_cfg["save_path"] = Path(bundle[SAVE_DIR_BUNDLE_KEY]) / self.tag

        vocabulary = beaver.get_vocabulary(dataset=df, **call_cfg)

        # store under this block’s namespace
        bundle[f"{self.tag}.{self.provides[0]}"] = vocabulary
