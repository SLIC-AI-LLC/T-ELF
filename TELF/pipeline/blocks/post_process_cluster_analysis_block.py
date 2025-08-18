from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

import numpy as np
import pandas as pd

from .base_block import AnimalBlock
from .data_bundle import DataBundle, SAVE_DIR_BUNDLE_KEY
from ...post_processing.ArcticFox import ClusteringAnalyzer


class ClusteringAnalyzerBlock(AnimalBlock):
    """
    Wraps the ClusteringAnalyzer with conditional inputs based on a single `mode` parameter.

    always needs      : ('df',)
    conditional needs :
      - flat        : 'W','H','vocab'
      - hierarchical: 'hnmfk_model','vocab'
      - label       : none (uses `cluster_col` from call_settings)
      - passthrough : none
    provides          : ('result',)   → path or list of paths to generated CSV(s)
    tag               : 'ClusteringAnalyzer'
    """
    CANONICAL_NEEDS: Tuple[str, ...] = ("df",)
    VALID_MODES = ["nmf", "hnmf", "label", None]

    def __init__(
        self,
        *,
        mode: str | None = None,
        needs: Sequence[str] = CANONICAL_NEEDS,
        provides: Sequence[str] = ("clusters_path",),
        tag: str = "ClusteringAnalyzer",
        init_settings: Dict[str, Any] | None = None,
        call_settings: Dict[str, Any] | None = None,
        **kw: Any,
    ) -> None:
        if mode not in self.VALID_MODES:
            raise ValueError(f"Invalid mode '{mode}'. Valid modes: {self.VALID_MODES}")
        self.mode = mode

        # shared init settings for ClusteringAnalyzer
        default_init = {
            "top_n_words": 50,
            "out_dir": None,
            "archive_subdir": "archive",
            "default_clean_col": "clean_title_abstract",
            "wordcloud_size": (800, 800),
            "max_font_size": 80,
            "contour_width": 1,
            "col_year": "year",
            "col_type": "type",
            "col_cluster": "cluster",
            "col_cluster_coords": "cluster_coordinates",
            "col_similarity": "similarity_to_cluster_centroid",
            "table_filename": "table_H-clustering.csv",
            "cluster_doc_map_filename": "cluster_documents_map.txt",
            "top_words_filename": "top_words.csv",
            "probs_filename": "probabilities_top_words.csv",
            "clusters_info_filename": "clusters_information.csv",
            "documents_info_filename": "documents_information.csv",
        }
        default_call = {
            "cluster_col": None,
            "clean_cols_name": None,
            "process_parents": True,
            "skip_completed": True,
        }

        init_cfg = {**default_init, **(init_settings or {})}
        call_cfg = {**default_call, **(call_settings or {})}

        # build conditional_needs based on mode
        conds: list[Tuple[str, Any]] = []
        if self.mode and self.mode  == self.VALID_MODES[0]:
            conds = [
                ("nmfk_model", lambda b, s: True),
                ("nmfk_model_path", lambda b, s: True),
                ("vocabulary", lambda b, s: True),
            ]
        elif self.mode and self.mode == self.VALID_MODES[1]:
            conds = [
                ("hnmfk_model", lambda b, s: True),
                ("vocabulary", lambda b, s: True),
            ]
        # 'label' and 'passthrough' modes require no extra bundle keys;
        # label-based logic uses call_settings['cluster_col'] at runtime

        super().__init__(
            needs=needs,
            provides=provides,
            conditional_needs=conds,
            tag=tag,
            init_settings=init_cfg,
            call_settings=call_cfg,
            **kw,
        )

    def run(self, bundle: DataBundle) -> None:
        # 1) load the input DataFrame
        df_val = bundle[self.needs[0]]
        df = self.load_path(df_val) if isinstance(df_val, (str, Path)) else df_val

        # 2) instantiate the analyzer
        analyzer = ClusteringAnalyzer(**self.init_settings)
        result = None

        # dispatch based on mode
        if not self.mode:
            if not self.init_settings.get("out_dir"):
                analyzer.out_dir = Path(bundle[SAVE_DIR_BUNDLE_KEY]) / self.tag
            result = analyzer.analyze(df)

        elif self.mode == self.VALID_MODES[1]:

            h_model = bundle["hnmfk_model"]
            vocab   = bundle["vocabulary"]

            if not self.init_settings.get("out_dir"):
                analyzer.out_dir = str(Path(h_model.experiment_save_path))

            result = analyzer.analyze(
                df,
                hnmfk_model     = h_model,
                vocab           = vocab,
                clean_cols_name = self.call_settings["clean_cols_name"],
                process_parents = self.call_settings["process_parents"],
                skip_completed  = self.call_settings["skip_completed"],
            )
        elif self.mode == self.VALID_MODES[0] :
            nmfk_model = bundle['nmfk_model'] 
            nmfk_model_path = bundle['nmfk_model_path'] 


            if not self.init_settings.get("out_dir"):
                analyzer.out_dir = str(Path(nmfk_model_path).parent)

            W, H = nmfk_model['W'], nmfk_model['H']

            vocab = bundle["vocabulary"]
            result = analyzer.analyze(
                df,
                W     = W,
                H     = H,
                vocab = vocab,
            )
        elif self.mode == self.VALID_MODES[2] :
            if not self.init_settings.get("out_dir"):
                analyzer.out_dir = Path(bundle[SAVE_DIR_BUNDLE_KEY]) / self.tag
            result = analyzer.analyze(
                df,
                cluster_col = self.call_settings["cluster_col"],
            )

        # 4) register the output
        bundle[f"{self.tag}.{self.provides[0]}"] = result
