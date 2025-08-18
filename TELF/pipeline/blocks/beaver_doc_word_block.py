# blocks/beaver_docword_block.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence, Any

from ...pre_processing.Beaver import Beaver
from .base_block import AnimalBlock
from .data_bundle import DataBundle,SAVE_DIR_BUNDLE_KEY


class BeaverDocWordBlock(AnimalBlock):
    """
    Build a sparse **document-word** matrix with *Beaver*.

    ─────────────────────────────────────────────────────────────
    needs        : ('df', 'vocabulary')
    provides     : ('X',)                      – the matrix
    tag          : 'BeaverDW'
    """
    CANONICAL_NEEDS = ("df", "vocabulary", )

    def __init__(
        self,
        *,
        col: str = "clean_title_abstract",
        needs: Sequence[str] = CANONICAL_NEEDS,
        provides: Sequence[str] = ("X",),
        conditional_needs: Sequence[tuple[str, Any]] = (),   # none for now
        tag: str = "BeaverDW",
        init_settings: Dict[str, Any] | None = None,
        call_settings: Dict[str, Any] | None = None,
        **kw,
    ) -> None:

        self.col = col
        default_init = {
            "n_jobs": -1,
            "n_nodes": 1,
        }

        default_call = {
            'target_column':self.col,
            'options':{"min_df": 5, "max_df": 0.5},
            'highlighting':[],
            'weights':[],
            'matrix_type':"tfidf",
            'verbose':False,
            'return_object':True,
            'output_mode':'scipy',
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

    # ------------------------------------------------------------------ #
    # work                                                                #
    # ------------------------------------------------------------------ #
    def run(self, bundle: DataBundle) -> None:
        # fetch dataframe
        src_df = bundle[self.needs[0]]
        if isinstance(src_df, (str, Path)):
            df = self.load_path(src_df)
        else:
            df = src_df.copy()

        # fetch vocabulary (could be namespaced or generic)
        vocab = bundle[self.needs[1]]

        # prepare dataframe
        target = self.call_settings["target_column"]
        df = df.dropna(subset=[target])

        # build doc-word matrix
        beaver = Beaver(**self.init_settings)

        cb = dict(self.call_settings)
        if SAVE_DIR_BUNDLE_KEY in bundle:
            out_dir = Path(bundle[SAVE_DIR_BUNDLE_KEY]) / self.tag
            out_dir.mkdir(parents=True, exist_ok=True)
            cb.setdefault("save_path", out_dir)
        
        cb["options"]["vocabulary"] = vocab
        X, _ = beaver.documents_words(dataset=df, **cb) 
        X = X.T.tocsr()
        # store matrix under this block’s namespace
        bundle[f"{self.tag}.{self.provides[0]}"] = X
