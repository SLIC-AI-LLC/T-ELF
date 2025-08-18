from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence, Any, Tuple

from .base_block import AnimalBlock
from .data_bundle import DataBundle, SAVE_DIR_BUNDLE_KEY
from ...factorization import NMFk


_NEEDS_MASK = {"nmf_recommender", "wnmf", "bnmf"}   # methods that require a mask


class NMFkBlock(AnimalBlock):
    """
    Wrapper around **NMFk**.

        always needs  'X'
        needs  'MASK' only when  init_settings['nmf_method']  ∈ {_NEEDS_MASK}
        provides   : ("results", "model_path", )
    """
    CANONICAL_NEEDS = ("X", )

    def __init__(
        self,
        *,
        needs: Sequence[str] = CANONICAL_NEEDS,
        provides: Sequence[str] = ("nmfk_model", "nmfk_model_path"  ),
        tag: str = "NMFk",
        conditional_needs: Sequence[Tuple[str, Any]] | None = None,
        init_settings: Dict[str, Any] | None = None,
        call_settings: Dict[str, Any] | None = None,
        **kw,
    ) -> None:

        default_init = {
            # ---------- NMFk defaults ----------
            "n_perturbs": 5,
            "n_iters": 500,
            "epsilon": 0.015,
            "n_jobs": -1,
            "init": "nnsvd",
            "use_gpu": False,
            "save_output": True,
            "collect_output": True,
            "predict_k_method": "sill",
            "verbose": True,
            "nmf_verbose": False,
            "transpose": False,
            "sill_thresh": 0.8,
            "pruned": True,
            "nmf_method": "nmf_fro_mu",          # ← drives MASK requirement
            "calculate_error": True,
            "predict_k": True,
            "use_consensus_stopping": 0,
            "calculate_pac": True,
            "consensus_mat": True,
            "perturb_type": "uniform",
            "perturb_multiprocessing": False,
            "perturb_verbose": False,
            "simple_plot": True,
            "k_search_method": "bst_pre",
            "H_sill_thresh": 0.1,
            "clustering_method": "kmeans",
            "device": -1,
            "nmf_obj_params": {},
        }

        default_call = {
            "Ks": range(1, 21),
            "name": "Example_NMFk",
            "note": "This is an example run of NMFk",
        }

        init_cfg = {**default_init, **(init_settings or {})}

        # ------------------------------------------------------------------
        # conditional MASK rule: *register it only if nmf_method needs one*
        # ------------------------------------------------------------------
        conds = list(conditional_needs or [])
        if init_cfg["nmf_method"] in _NEEDS_MASK:
            # an always-true condition is enough because we add it ONLY
            # in the branch that needs it
            conds.append(("MASK", lambda _b, _s: True))

        super().__init__(
            needs=needs,
            provides=provides,
            conditional_needs=conds,
            tag=tag,
            init_settings=self._merge(init_cfg, init_settings),
            call_settings=self._merge(default_call, call_settings),
            **kw,
        )

    # ------------------------------------------------------------------ #
    # run                                                                #
    # ------------------------------------------------------------------ #
    def run(self, bundle: DataBundle) -> None:
        # 1  — load X
        X_val = bundle[self.needs[0]]
        X = self.load_path(X_val) if isinstance(X_val, (str, Path)) else X_val

        if self.init_settings["nmf_method"] in _NEEDS_MASK:
            self.init_settings["nmf_obj_params"]["MASK"] = bundle["MASK"]

        init_cfg = dict(self.init_settings)
        if SAVE_DIR_BUNDLE_KEY not in init_cfg and SAVE_DIR_BUNDLE_KEY in bundle:
            init_cfg[SAVE_DIR_BUNDLE_KEY] = Path(bundle[SAVE_DIR_BUNDLE_KEY]) / self.tag

        model = NMFk(**init_cfg)
        results = model.fit(X, **self.call_settings)

        # 4  — store
        bundle[f"{self.tag}.{self.provides[0]}"] = results
        bundle[f"{self.tag}.{self.provides[1]}"] = model.save_path_full

    def _after_checkpoint_skip(self, bundle: DataBundle) -> None:
        """Re-insert fast-to-make steps after checkpoint reload."""
        # steps = self.__rebuild_steps(bundle)
        # bundle[f"{self.tag}.{self.provides[1]}"] = steps