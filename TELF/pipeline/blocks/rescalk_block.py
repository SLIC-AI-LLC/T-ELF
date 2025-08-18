# blocks/rescalc_block.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

from ...factorization import RESCALk
from .base_block import AnimalBlock
from .data_bundle import DataBundle, SAVE_DIR_BUNDLE_KEY


class RESCALkBlock(AnimalBlock):
    """
    Pipeline wrapper around **RESCALk** (tensor factorisation).

    needs
    -----
    ('X',)              – a 3-mode tensor (NumPy ndarray, sparse COO, or on-disk path)

    provides
    --------
    ('results', 'model_path')

        * **results**     – output of :py:meth:`RESCALk.fit`
        * **model_path**  – directory where the model artefacts were saved

    tag
    ---
    'RESCALk'
    """

    CANONICAL_NEEDS = ("X",)

    # ------------------------------------------------------------------ #
    # constructor                                                         #
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        *,
        needs: Sequence[str] = CANONICAL_NEEDS,
        provides: Sequence[str] = ("results", "model_path"),
        conditional_needs: Sequence[Tuple[str, Any]] = (),
        tag: str = "RESCALk",
        init_settings: Dict[str, Any] | None = None,
        call_settings: Dict[str, Any] | None = None,
        verbose: bool = True,
        **kwargs: Any,
    ) -> None:

        # ------------------- default constructor params ------------------- #
        default_init = {
            "n_perturbs": 12,
            "n_iters": 50,
            "epsilon": 0.015,
            "n_jobs": -1,
            "n_nodes": 1,
            "init": "nnsvd",
            "use_gpu": True,
            # save_path will be injected automatically if not supplied
            "save_output": True,
            "verbose": True,
            "pruned": False,
            "rescal_verbose": False,
            "calculate_error": True,
            "rescal_func": None,
            "rescal_obj_params": {},
            "simple_plot": True,
            "rescal_method": "rescal_fro_mu",
            "get_plot_data": True,
            "perturb_type": "uniform",
            "perturb_multiprocessing": False,
            "perturb_verbose": False,
        }

        # ------------------- default arguments to `.fit()` ---------------- #
        default_call = {
            "Ks": range(1, 7),             # 1 … 6
            "name": "RESCALk",
            "note": "This is an example run of RESCALk",
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
        # 1️⃣  Load tensor X --------------------------------------------------
        src_X = bundle[self.needs[0]]
        X = self.load_path(src_X) if isinstance(src_X, (str, Path)) else src_X

        # 2️⃣  Ensure `save_path` exists / is injected -----------------------
        init_cfg: Dict[str, Any] = dict(self.init_settings)
        if SAVE_DIR_BUNDLE_KEY not in init_cfg and SAVE_DIR_BUNDLE_KEY in bundle:
            init_cfg[SAVE_DIR_BUNDLE_KEY] = Path(bundle[SAVE_DIR_BUNDLE_KEY]) / self.tag

        # 3️⃣  Fit RESCALk ----------------------------------------------------
        model = RESCALk(**init_cfg)
        results = model.fit(
            X,
            self.call_settings["Ks"],
            self.call_settings["name"],
            self.call_settings["note"],
        )

        # 4️⃣  Store outputs in the bundle -----------------------------------
        ns = self.tag
        bundle[f"{ns}.{self.provides[0]}"] = results
        bundle[f"{ns}.{self.provides[1]}"] = model.save_path_full
