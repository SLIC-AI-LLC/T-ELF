# blocks/compute_scores_block.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Sequence, Tuple, Optional

import numpy as np
import pandas as pd
from .base_block import AnimalBlock
from .data_bundle import DataBundle

from sklearn.metrics import (
    root_mean_squared_error,
    f1_score,
    precision_score,
    recall_score,
    auc,
    accuracy_score,
    matthews_corrcoef,
    precision_recall_curve,
    roc_curve,
)

# ------------------------------------------------------------------ #
# helper                                                             #
# ------------------------------------------------------------------ #
def _compute_scores(
    y_true: np.ndarray | pd.Series | list,
    y_pred: np.ndarray | pd.Series | list,
    *,
    binary: bool = True,
) -> Dict[str, float | None]:
    """
    Replicates the original `get_scores` function and returns a dict of metrics.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # ROC / PR curves expect probabilities; we fall back to labels gracefully.
    fpr, tpr, _ = roc_curve(y_true, y_pred, pos_label=1)
    roc_auc = auc(fpr, tpr)

    precision_line, recall_line, _ = precision_recall_curve(
        y_true, y_pred, pos_label=1
    )
    pr_auc = auc(recall_line, precision_line)

    rmse = root_mean_squared_error(y_true, y_pred)

    if binary:
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="weighted")
        precision = precision_score(y_true, y_pred, average="weighted")
        recall = recall_score(y_true, y_pred, average="weighted")
        mcc = matthews_corrcoef(y_true, y_pred)
    else:
        acc = f1 = precision = recall = mcc = None

    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "mcc": mcc,
        "rmse": rmse,
    }


# ------------------------------------------------------------------ #
# block                                                              #
# ------------------------------------------------------------------ #
class ComputeScoresBlock(AnimalBlock):
    """
    Compute evaluation metrics for model predictions.

    needs        : ('y_true', 'y_pred',)
    provides     : ('scores', 'scores_df',)
    tag          : 'ComputeScores'
    conditional  : none
    """
    CANONICAL_NEEDS = ("y_true", "y_pred",)

    def __init__(
        self,
        *,
        needs: Sequence[str] = CANONICAL_NEEDS,
        provides: Sequence[str] = ("scores",),
        conditional_needs: Sequence[Tuple[str, Any]] = (),
        tag: str = "ComputeScores",
        init_settings: Dict[str, Any] | None = None,
        call_settings: Dict[str, Any] | None = None,
        verbose: bool = True,
        **kwargs: Any,
    ) -> None:

        default_init = {"verbose": True}
        default_call = {
            "binary": True,
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
        # Pull y_true and y_pred (either paths or in-memory arrays/Series)
        y_true = self.load_path(bundle[self.needs[0]])
        y_pred = self.load_path(bundle[self.needs[1]])

        # Compute scores
        scores_dict = _compute_scores(y_true, y_pred, **self.call_settings)

        # Store: as dict and as single-row DataFrame for convenience
        bundle[f"{self.tag}.{self.provides[0]}"] = scores_dict
        bundle[f"{self.tag}.{self.provides[0]}_df"] = pd.DataFrame([scores_dict])
