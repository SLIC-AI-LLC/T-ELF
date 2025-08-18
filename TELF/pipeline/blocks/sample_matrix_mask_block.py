# blocks/sample_matrix_block.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .base_block import AnimalBlock
from .data_bundle import DataBundle


# ------------------------------------------------------------------ #
# helper                                                             #
# ------------------------------------------------------------------ #
def _sample_matrix(
    X: np.ndarray,
    *,
    sample_ratio: float = 0.05,
    random_state: int = 42,
    stratify: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Re-implements the original `sample_matrix` helper.

    Returns
    -------
    X_orig         : original matrix (float32)
    X_train        : matrix with sampled entries set to 0 (float32)
    mask           : binary mask (1 = kept, 0 = removed)  (int8)
    removed_coords : coordinates of removed elements, shape (N, 2)  (int32)
    """
    flat_matrix = X.flatten()
    indices = np.arange(flat_matrix.size)

    # choose indices to remove
    if stratify:
        _, sampled_idx = train_test_split(
            indices,
            test_size=sample_ratio,
            stratify=flat_matrix,
            random_state=random_state,
        )
    else:
        _, sampled_idx = train_test_split(
            indices,
            test_size=sample_ratio,
            random_state=random_state,
        )

    # create masked copy
    X_train = X.astype(float).copy()
    np.put(X_train, sampled_idx, np.nan)

    removed_coords = np.argwhere(np.isnan(X_train))
    mask = np.ones_like(X_train, dtype=int)
    mask[tuple(removed_coords.T)] = 0
    X_train[np.isnan(X_train)] = 0  # replace NaN so downstream code is safe

    return (
        X.astype("float32"),
        X_train.astype("float32"),
        mask.astype("int8"),
        removed_coords.astype("int32"),
    )


# ------------------------------------------------------------------ #
# block                                                              #
# ------------------------------------------------------------------ #
class SampleMatrixMaskBlock(AnimalBlock):
    """
    Randomly masks a proportion of entries in a binary matrix.

    needs        : ('X',)
    provides     : ('X_train', 'MASK', 'removed_coords')
    tag          : 'SampleMatrix'
    """

    CANONICAL_NEEDS = ("X",)

    def __init__(
        self,
        *,
        needs: Sequence[str] = CANONICAL_NEEDS,
        provides: Sequence[str] = ("X_train", "MASK", "removed_coords",),
        conditional_needs: Sequence[Tuple[str, Any]] = (),
        tag: str = "SampleMatrix",
        init_settings: Dict[str, Any] | None = None,
        call_settings: Dict[str, Any] | None = None,
        verbose: bool = True,
        **kwargs: Any,
    ) -> None:

        default_init = {"verbose": True}
        default_call = {
            "sample_ratio": 0.05,
            "random_state": 42,
            "stratify": True,
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
        #  Load matrix (path → DataFrame/ndarray or in-memory)
        X = self.load_path(bundle[self.needs[0]])

        # Sample / mask
        X_orig, X_train, mask, removed_coords = _sample_matrix(X, **self.call_settings)

        # Store outputs under this block’s namespace
        ns = self.tag
        bundle[f"{ns}.X"] = X_orig
        bundle[f"{ns}.X_train"] = X_train
        bundle[f"{ns}.MASK"] = mask
        bundle[f"{ns}.removed_coords"] = removed_coords
