# blocks/beaver_something_words_block.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

from ...pre_processing.Beaver import Beaver
from .base_block import AnimalBlock
from .data_bundle import DataBundle, SAVE_DIR_BUNDLE_KEY


class BeaverSomethingWordsBlock(AnimalBlock):
    """
    Build a sparse **something-words** matrix with *Beaver*.

    ─────────────────────────────────────────────────────────────
    needs        : ('df', 'vocabulary')
    provides     : ('X', 'keys', 'vocabulary')   – matrix, ids, new vocab
    tag          : 'BeaverSW'
    """
    CANONICAL_NEEDS = ("df", "vocabulary")

    # ------------------------------------------------------------------ #
    # constructor                                                         #
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        *,
        target_columns: tuple[str, str] = ("author_ids", "clean_abstract"),
        needs: Sequence[str] = CANONICAL_NEEDS,
        provides: Sequence[str] = ("X", "keys", "vocabulary"),
        conditional_needs: Sequence[Tuple[str, Any]] = (),
        tag: str = "BeaverSW",
        init_settings: Dict[str, Any] | None = None,
        call_settings: Dict[str, Any] | None = None,
        verbose: bool = True,
        **kwargs: Any,
    ) -> None:

        # Settings forwarded to Beaver(...)
        default_init = {
            "n_jobs": -1,
            "n_nodes": 1,
        }

        # Settings forwarded to beaver.something_words(...)
        default_call = {
            "target_columns": target_columns,
            "options": {"min_df": 5, "max_df": 0.5},   # vocabulary injected at run-time
            "split_something_with": ";",
            "matrix_type": "tfidf",
            "highlighting": [],
            "weights": [],
            "verbose": False,
            "return_object": True,
            "output_mode": "scipy",
            # "save_path" will be added automatically if bundle["result_path"] exists
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
        # 1️⃣  Fetch dataframe ------------------------------------------------
        src_df = bundle[self.needs[0]]
        df = self.load_path(src_df) if isinstance(src_df, (str, Path)) else src_df.copy()

        # 2️⃣  Fetch vocabulary ----------------------------------------------
        vocab = bundle[self.needs[1]]

        # 3️⃣  Clean dataframe (drop rows with missing text columns) ----------
        for col in self.call_settings["target_columns"]:
            if col in df.columns:
                df = df.dropna(subset=[col])

        # 4️⃣  Prepare Beaver & call spec ------------------------------------
        beaver = Beaver(**self.init_settings)

        cb: Dict[str, Any] = dict(self.call_settings)

        # inject vocabulary
        cb.setdefault("options", {})
        cb["options"]["vocabulary"] = vocab

        # dynamic save path
        if "save_path" not in cb and SAVE_DIR_BUNDLE_KEY in bundle:
            cb["save_path"] = Path(bundle[SAVE_DIR_BUNDLE_KEY]) / self.tag

        # build the matrix
        X, keys, vocab_new = beaver.something_words(dataset=df, **cb)

        # 5️⃣  Store under namespaced keys ------------------------------------
        ns = self.tag
        bundle[f"{ns}.{self.provides[0]}"] = X            # 'X'
        bundle[f"{ns}.{self.provides[1]}"] = keys         # 'keys' (e.g., author_ids)
        bundle[f"{ns}.{self.provides[2]}"] = vocab_new    # updated vocabulary
