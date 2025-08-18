# blocks/codependency_matrix_block.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

import os, sparse
from ...pre_processing import Beaver
from ...helpers.file_system import load_file_as_dict

from .base_block import AnimalBlock
from .data_bundle import DataBundle, SAVE_DIR_BUNDLE_KEY


class CodependencyMatrixBlock(AnimalBlock):
    """
    Build a 3-mode author–year tensor and flatten it to a co-authorship
    matrix + node-ID map.

    ─────────────────────────────────────────────────────────────
    needs        : ('df',)
    provides     : ('X', 'node_ids')
    tag          : 'CodeMatrix'
    """
    CANONICAL_NEEDS = ('df', )

    # ------------------------------------------------------------------ #
    # constructor                                                        #
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        *,
        col: str = "slic_author_ids",
        needs: Sequence[str] = CANONICAL_NEEDS,
        provides: Sequence[str] = ("X", "node_ids"),
        conditional_needs: Sequence[Tuple[str, Any]] = (),   # none for now
        tag: str = "CodeMatrix",
        init_settings: Dict[str, Any] | None = None,
        call_settings: Dict[str, Any] | None = None,
        verbose: bool = True,
        **kwargs: Any,
    ) -> None:

        self.col = col  # store the column name

        default_init = {}
        default_call = {
            "target_columns": [self.col, "year"],
            "split_authors_with": ";",
            "verbose": True,
            "n_jobs": -1,
            "authors_idx_map": {},
            "joblib_backend": "threading",
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
        # paths
        out_dir = Path(bundle[SAVE_DIR_BUNDLE_KEY]) / "CodependencyMatrixBlock" / self.col
        out_dir.mkdir(parents=True, exist_ok=True)

        # dataframe
        df = bundle[self.needs[0]].copy()

        # build tensor with Beaver
        beaver = Beaver(**self.init_settings)
        cfg = dict(self.call_settings)
        cfg.update({"dataset": df, "target_columns": [self.col, "year"], "save_path": out_dir})

        beaver.coauthor_tensor(**cfg)

        # load results
        X = sparse.load_npz(out_dir / "coauthor.npz").sum(axis=2)     # flatten 3-mode tensor
        node_ids = load_file_as_dict(out_dir / "Authors.txt")

        # write back under this block’s namespace
        bundle[f"{self.tag}.{self.provides[0]}"] = X
        bundle[f"{self.tag}.{self.provides[1]}"] = node_ids
