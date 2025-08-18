# blocks/hnmfk_block.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

from .base_block import AnimalBlock
from .data_bundle import DataBundle, SAVE_DIR_BUNDLE_KEY
from ...factorization import HNMFk
from copy import deepcopy

class HNMFkBlock(AnimalBlock):
    CANONICAL_NEEDS = ("X", )

    def __init__(
        self,
        *,
        needs: Sequence[str] = CANONICAL_NEEDS,
        provides: Sequence[str] = ("hnmfk_model", "saved_path"),
        conditional_needs: Sequence[Tuple[str, Any]] = (),
        tag: str = "HNMFk",
        init_settings: Dict[str, Any] | None = None,
        call_settings: Dict[str, Any] | None = None,
        verbose: bool = True,
        **kw: Any,
    ) -> None:
        # 1) your single default nmfk_params dict:
        base_nmfk = {
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
            "nmf_method": "nmf_fro_mu",
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
        }

        # 2) build a clean copy of default_init
        default_init = {
            "nmfk_params": [base_nmfk],
            "cluster_on": "H",
            "depth": 1,
            "sample_thresh": 5,
            "K2": False,
            "Ks_deep_min": 1,
            "Ks_deep_max": 20,
            "Ks_deep_step": 1,
            "random_identifiers": False,
            "root_node_name": "Root",
        }

        # 3) if the user passed nmfk_params in init_settings, special-case it:
        user_init = deepcopy(init_settings) or {}
        if "nmfk_params" in user_init:
            overrides = user_init.pop("nmfk_params")
            # make sure we have a list of dicts
            if isinstance(overrides, dict):
                overrides = [overrides]
            # for each override dict, shallow-merge it onto a fresh base_nmfk
            merged_list = []
            for ov in overrides:
                merged = {**base_nmfk, **ov}
                merged_list.append(merged)
            default_init["nmfk_params"] = merged_list

        # 4) now do your normal (shallow) merge for everything else
        def _merge(default: Dict[str, Any], override: Dict[str, Any] | None):
            return {**default, **(override or {})}

        merged_init = _merge(default_init, user_init)

        # 5) build call_settings as before
        default_call = {
            "Ks": range(1, 21),
            "from_checkpoint": True,
            "save_checkpoint": True,
        }
        merged_call = _merge(default_call, call_settings)

        # 6) finally hand everything up to the base class
        super().__init__(
            needs=needs,
            provides=provides,
            conditional_needs=conditional_needs,
            tag=tag,
            init_settings=merged_init,
            call_settings=merged_call,
            verbose=verbose,
            **kw,
        )

    # ------------------------------------------------------------------ #
    # work                                                                #
    # ------------------------------------------------------------------ #
    def run(self, bundle: DataBundle) -> None:
        # load matrix X
        src = bundle[self.needs[0]]
        X = self.load_path(src) if isinstance(src, (str, Path)) else src

        init_cfg = dict(self.init_settings)
        if "experiment_name" not in init_cfg and SAVE_DIR_BUNDLE_KEY in bundle:
            init_cfg["experiment_name"] = Path(bundle[SAVE_DIR_BUNDLE_KEY]) / self.tag

        # fit hierarchical model
        model = HNMFk(**init_cfg)
        model.fit(X, **self.call_settings)

        # store under namespace
        bundle[f"{self.tag}.{self.provides[0]}"] = model
        bundle[f"{self.tag}.{self.provides[1]}"] = model.experiment_save_path
