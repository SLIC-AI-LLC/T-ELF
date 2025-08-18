from pathlib import Path
from typing import Dict, Sequence, Any
import pickle
from .base_block import AnimalBlock
from .data_bundle import DataBundle, SAVE_DIR_BUNDLE_KEY

from ...factorization.decompositions.lmf import LogisticMatrixFactorization

class LMFBlock(AnimalBlock):
    CANONICAL_NEEDS = ("X", "MASK",)

    def __init__(
        self,
        *,
        threshold:float=0.5,
        needs = CANONICAL_NEEDS,
        provides = ("W", "H", "row_bias", "col_bias", "losses", "Xtilda", "Xtilda_bool", "model",),
        conditional_needs: Sequence[tuple[str, Any]] = (),   # none today
        tag: str = "LMF",
        init_settings: Dict[str, Any] | None = None,
        call_settings: Dict[str, Any] | None = None,
        **kw,
    ):
        """
        A component for performing low-rank matrix factorization with masking.

        This module is designed to work within a pipeline framework where it declares
        its input needs and what it provides after execution.

        Parameters:
        ----------
        threshold : float, default=0.5
            A threshold value used during internal computations (e.g., for binary masking or evaluation).
        needs : tuple of str, default=("X", "MASK",)
            The names of inputs this module requires.
        provides : tuple of str, default=("W", "H", "row_bias", "col_bias", "losses", "Xtilda", "Xtilda_bool", "model")
            The names of outputs this module will produce and provide.
        conditional_needs : Sequence[tuple[str, Any]], default=()
            Additional inputs needed conditionally based on runtime logic (unused currently).
        tag : str, default="LMF"
            A label used to tag or identify the module in logs or output.
        init_settings : dict, optional
            Additional settings used during initialization.
        call_settings : dict, optional
            Settings to control behavior during the execution or "call" phase.
        **kw : dict
            Arbitrary additional keyword arguments.

        Needs:
        ------
        - "X": Input data matrix.
        - "MASK": A mask matrix indicating observed vs. missing entries.

        Provides:
        ---------
        - "W": Left factor matrix.
        - "H": Right factor matrix.
        - "row_bias": Bias values for rows.
        - "col_bias": Bias values for columns.
        - "losses": Loss values recorded during training.
        - "Xtilda": Reconstructed matrix.
        - "Xtilda_bool": Binary thresholded version of the reconstruction.
        - "model": The trained model or relevant object representing it.
        """
        self.threshold=threshold
        default_init = {
            "k": 5,
            "l2_p": 1e-6,
            "epochs": 3000,
            "learning_rate": 0.001,
            "tolerance": 1e-3,
            "device": 2,
            "random_state": 1
        }
        default_call = {
            "plot_loss":False
        }

        super().__init__(
            needs=needs,
            provides=provides,
            conditional_needs=conditional_needs,
            tag=tag,
            init_settings=self._merge(default_init, init_settings),
            call_settings=self._merge(default_call, call_settings),
            **kw,
        )


    def run(self, bundle: DataBundle) -> None:
        # 1  — load X
        X = self.load_path(bundle[self.needs[0]])
        MASK = self.load_path(bundle[self.needs[1]])

        if SAVE_DIR_BUNDLE_KEY in bundle:
            path = Path(bundle[SAVE_DIR_BUNDLE_KEY]) / self.tag
        else:
            path = self.tag

        model = LogisticMatrixFactorization(**self.init_settings)
        W, H, row_bias, col_bias, losses = model.fit(X, MASK, **self.call_settings)
        Xtilda = model.predict(W, H, row_bias, col_bias)
        Xtilda_bool = model.map_probabilities_to_binary(Xtilda, threshold=self.threshold)

        pickle.dump(
            {"W":W, 
             "H":H, 
             "row_bias":row_bias, "col_bias":col_bias, 
             "losses":losses,
             "Xtilda":Xtilda,
             "Xtilda_bool":Xtilda_bool}, 
            open(path, "wb"))

        # 4  — store
        bundle[f"{self.tag}.{self.provides[0]}"] = W
        bundle[f"{self.tag}.{self.provides[1]}"] = H
        bundle[f"{self.tag}.{self.provides[2]}"] = row_bias
        bundle[f"{self.tag}.{self.provides[3]}"] = col_bias
        bundle[f"{self.tag}.{self.provides[4]}"] = losses
        bundle[f"{self.tag}.{self.provides[5]}"] = Xtilda
        bundle[f"{self.tag}.{self.provides[6]}"] = Xtilda_bool
        bundle[f"{self.tag}.{self.provides[7]}"] = model
