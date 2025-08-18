# blocks/split_transfer_block.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

import numpy as np
import pandas as pd

from .base_block import AnimalBlock
from .data_bundle import DataBundle, SAVE_DIR_BUNDLE_KEY
from ...factorization import SPLITTransfer

class SPLITTransferBlock(AnimalBlock):
    """
    Pipeline block that wraps :class:`TELF.factorization.SPLITTransfer`.

    needs
    -----
    ('X_known', 'X_target', 'indicator')

        * **X_known**   – source-domain matrix (sparse or dense, or a path)
        * **X_target**  – target-domain matrix (same format as above)
        * **indicator** – boolean / 0-1 array indicating the held-out
                          positions to predict in *X_target*

    provides
    --------
    ('model', 'y_pred_train', 'y_pred_test')

        * **model**         – fitted :class:`SPLITTransfer`
        * **y_pred_train**  – predictions (source/target) for the *train* mask
        * **y_pred_test**   – predictions for the *test* mask

    tag
    ---
    'SPLITTransfer'
    """

    CANONICAL_NEEDS = ("X_known", "X_target", "indicator")

    # ------------------------------------------------------------------ #
    # constructor                                                         #
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        *,
        needs: Sequence[str] = CANONICAL_NEEDS,
        provides: Sequence[str] = ("model", "y_pred_train", "y_pred_test"),
        conditional_needs: Sequence[Tuple[str, Any]] = (),
        tag: str = "SPLITTransfer",
        init_settings: Dict[str, Any] | None = None,
        call_settings: Dict[str, Any] | None = None,
        verbose: bool = True,
        **kwargs: Any,
    ) -> None:

        # ------------------- base hyper-parameters -------------------- #
        _base_known = {
            "n_perturbs": 30,
            "n_iters": 1000,
            "epsilon": 0.015,
            "n_jobs": -1,
            "init": "nnsvd",
            "use_gpu": False,
            "save_path": None,
            "save_output": True,
            "sill_thresh": 0.5,
            "nmf_method": "nmf_fro_mu",
        }
        _base_target = {**_base_known, "epsilon": 0.01, "sill_thresh": 0.8}
        _base_split = {**_base_known, "sill_thresh": 0.6}

        default_init = {
            # search ranges
            "Ks_known": range(1, 10),         # 1 … 9
            "Ks_target": range(1, 10),
            "Ks_split_step": 1,
            "Ks_split_min": 1,
            # regression hyper-params
            "H_regress_gpu": False,
            "H_learn_method": "regress",
            "H_regress_init": "MitH",
            "H_regress_iters": 1000,
            "H_regress_method": "fro",
            # nested NMFk parameters
            "nmfk_params_known": _base_known,
            "nmfk_params_target": _base_target,
            "nmfk_params_split": _base_split,
            # transfer-learning settings
            "transfer_method": "SVR",   # or "model"
            "transfer_regress_params": {},
            "transfer_model": None,     # supply if transfer_method == "model"
            # misc
            "verbose": True,
            "random_state": 42,
        }

        # forwarded to model.fit / transform / predict if ever needed
        default_call: Dict[str, Any] = {}

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
    def _to_numpy(self, obj: Any) -> np.ndarray:
        """Utility: convert DataFrame / Series to ndarray, keep ndarray else."""
        if isinstance(obj, (str, Path)):
            obj = self.load_path(obj)
        if isinstance(obj, pd.DataFrame):
            return obj.values
        if isinstance(obj, pd.Series):
            return obj.to_numpy()
        return np.asarray(obj)

    def run(self, bundle: DataBundle) -> None:
        # 1️⃣  Gather inputs ------------------------------------------------- #
        X_known_src, X_target_src, indicator_src = (
            bundle[n] for n in self.needs
        )

        X_known = self._to_numpy(X_known_src)
        X_target = self._to_numpy(X_target_src)
        indicator = self._to_numpy(indicator_src)

        # 2️⃣  Fit SPLITTransfer ------------------------------------------- #
        model = SPLITTransfer(**self.init_settings)
        model.fit(X_known, X_target)
        model.transform(indicator)

        # 3️⃣  Predictions -------------------------------------------------- #
        y_pred_train = model.predict(test=False)
        y_pred_test = model.predict(test=True)

        # 4️⃣  Store outputs ------------------------------------------------ #
        ns = self.tag
        bundle[f"{ns}.{self.provides[0]}"] = model
        bundle[f"{ns}.{self.provides[1]}"] = y_pred_train
        bundle[f"{ns}.{self.provides[2]}"] = y_pred_test
