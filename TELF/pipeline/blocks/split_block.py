# blocks/split_block.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

from .base_block import AnimalBlock
from .data_bundle import DataBundle
from ...factorization import SPLIT


class SPLITBlock(AnimalBlock):
    """
    Wrapper around TELF.factorization.SPLIT.

    needs
    -----
    ('Xs',)  – sparse or dense *collection* of matrices (see SPLIT docs)

    provides
    --------
    ('results', 'model')

        * **results** – output of `model.transform()`
        * **model**   – the fitted SPLIT instance itself

    tag
    ---
    'SPLIT'
    """

    CANONICAL_NEEDS = ("Xs",)

    # ------------------------------------------------------------------ #
    # constructor                                                         #
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        *,
        needs: Sequence[str] = CANONICAL_NEEDS,
        provides: Sequence[str] = ("results", "model"),
        conditional_needs: Sequence[Tuple[str, Any]] = (),
        tag: str = "SPLIT",
        init_settings: Dict[str, Any] | None = None,
        call_settings: Dict[str, Any] | None = None,
        verbose: bool = True,
        **kwargs: Any,
    ) -> None:

        # ---------- defaults that feed straight into SPLIT(...) -------- #
        params = {
            "n_perturbs": 10,
            "n_iters": 100,
            "epsilon": 0.015,
            "n_jobs": 1,
            "init": "nnsvd",
            "use_gpu": False,
            "save_output": True,
            "verbose": False,
            "transpose": False,
            "sill_thresh": 0.9,
            "nmf_method": "nmf_fro_mu",
        }

        Ks = {
            "X1": range(1, 9),
            "X2": range(1, 10),
            "X3": range(1, 11),
        }
        nmfk_params = {name: params for name in Ks}

        default_init = {
            "Ks": Ks,
            "nmfk_params": nmfk_params,
            "split_nmfk_params": params,
            "Ks_split_step": 1,
            "Ks_split_min": 1,
            "H_regress_gpu": False,
            "H_learn_method": "regress",
            "H_regress_iters": 1000,
            "H_regress_method": "fro",
            "H_regress_init": "MitH",
            "verbose": True,
            "random_state": 42,
        }

        # ---------- args to `model.fit(**call_settings)` ---------------- #
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
    # work                                                               #
    # ------------------------------------------------------------------ #
    def run(self, bundle: DataBundle) -> None:
        src = bundle[self.needs[0]]
        if isinstance(src, (str, Path)):
            Xs = self.load_path(src)
        else:
            Xs = src

        model = SPLIT(Xs=Xs, **self.init_settings)
        model.fit(**self.call_settings)

        results = model.transform()

        ns = self.tag
        bundle[f"{ns}.{self.provides[0]}"] = results
        bundle[f"{ns}.{self.provides[1]}"] = model
