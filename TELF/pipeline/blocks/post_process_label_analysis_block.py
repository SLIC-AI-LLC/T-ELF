# pipeline/blocks/label_analyzer_block.py
from __future__ import annotations

import re
from pathlib import Path
from typing   import Any, Dict, Iterable, Sequence, Tuple, Union

import pandas as pd

from .base_block  import AnimalBlock
from .data_bundle import DataBundle, SAVE_DIR_BUNDLE_KEY
from ...post_processing.ArcticFox.label_analyzer import LabelAnalyzer


# ─────────────────────────────────────────────────────────────────────────────
# helper – harvest all “cluster_for_k=*.csv” under a root folder
def _collect_csvs(root: Path) -> Iterable[Path]:
    pat = re.compile(r"cluster_for_k=\d+\.csv")
    for p in root.rglob("*.csv"):
        if pat.fullmatch(p.name):
            yield p.resolve()


class LabelAnalyzerBlock(AnimalBlock):
    """
    Generate concise topic labels using `LabelAnalyzer` for:

      • a single DataFrame / CSV
      • a list / tuple of CSV paths  (e.g., HNMF-k leaf outputs)
      • an entire directory tree containing `cluster_for_k=*.csv`

    provides → ('result',) : dict[csv_path → dict[int,str]]
               and         : list[str] of the written labels.csv files
    """

    CANONICAL_NEEDS: Tuple[str, ...] = ("clusters_path",)

    # --------------------------------------------------------------------- #
    def __init__(
        self,
        *,
        needs: Sequence[str] = CANONICAL_NEEDS,
        provides: Sequence[str] = ("result","label_paths"),
        tag: str = "LabelAnalyzer",
        init_settings: Dict[str, Any] | None = None,
        call_settings: Dict[str, Any] | None = None,
        **kw: Any,
    ) -> None:

        # ---------------- default configs -------------------------------- #
        default_init = dict(
            embedding_model="SCINCL",
            distance_metric="cosine",
            text_cols=["title", "abstract"],
            tfidf_top_k=12,
            include_bigrams=True,
        )
        default_call = dict(
            provider="ollama",
            model_name=None,
            openai_api_key=None,      # can also be supplied in the bundle
            cluster_strategy=None,    # auto-detect per-file
            cluster_col="cluster",
            top_n_words=20,
            num_candidates=12,
            use_gpu=False,
        )

        self.init_settings = {**default_init, **(init_settings or {})}
        self.call_settings = {**default_call, **(call_settings or {})}

        # conditional need for the OpenAI key
        conds = []
        if self.call_settings["provider"].lower() == "openai":
            conds = [("openai_api_key", lambda b, s: "openai_api_key" in b)]

        super().__init__(
            needs=needs,
            provides=provides,
            conditional_needs=conds,
            tag=tag,
            init_settings=self.init_settings,
            call_settings=self.call_settings,
            **kw,
        )

    # --------------------------------------------------------------------- #
    def _label_single_csv(self, csv_path: Path, labeler: LabelAnalyzer) -> Dict[int, str]:
        df = pd.read_csv(csv_path)

        # choose cluster strategy automatically
        strat = ("column" if self.call_settings["cluster_col"] in df.columns
                 else "single") if self.call_settings["cluster_strategy"] is None \
                 else self.call_settings["cluster_strategy"]

        labels = labeler.label_texts(
            df,
            provider         = self.call_settings["provider"],
            model_name       = self.call_settings["model_name"],
            openai_api_key   = self.call_settings["openai_api_key"],
            cluster_strategy = strat,
            cluster_col      = self.call_settings["cluster_col"],
            top_n_words      = self.call_settings["top_n_words"],
            num_candidates   = self.call_settings["num_candidates"],
            use_gpu          = self.call_settings["use_gpu"],
        )

        # write labels.csv next to input
        out_csv = csv_path.with_name("labels.csv")
        pd.DataFrame(sorted(labels.items()), columns=["cluster", "label"]).to_csv(out_csv, index=False)
        return labels

    # --------------------------------------------------------------------- #
    def run(self, bundle: DataBundle) -> None:  # imperative – no docstring lint
        src = bundle[self.needs[0]]

        # resolve inputs to an iterable of CSV files or a single DataFrame
        csv_paths: list[Path] = []
        dataframe_param: Union[pd.DataFrame, None] = None

        if isinstance(src, pd.DataFrame):
            dataframe_param = src
        elif isinstance(src, (str, Path)):
            p = Path(src)
            if p.is_dir():
                csv_paths = list(_collect_csvs(p))
            elif p.is_file():
                csv_paths = [p]
            else:
                raise FileNotFoundError(p)
        elif isinstance(src, (list, tuple)):
            csv_paths = [Path(s) for s in src]
        else:
            raise TypeError(f"Unsupported df input type {type(src)}")

        # out-dir fallback for temp files (single-DF case)
        base_out_dir = Path(bundle[SAVE_DIR_BUNDLE_KEY]) / self.tag
        base_out_dir.mkdir(parents=True, exist_ok=True)

        labeler = LabelAnalyzer(**self.init_settings)
        result_dict: Dict[str, Dict[int, str]] = {}
        written_csvs: list[str] = []

        # ---------------------------------- many CSVs --------------------- #
        if csv_paths:
            for csv_p in csv_paths:
                labels = self._label_single_csv(csv_p, labeler)
                result_dict[str(csv_p)] = labels
                written_csvs.append(str(csv_p.with_name("labels.csv")))

        # ---------------------------------- single DataFrame -------------- #
        if dataframe_param is not None:
            # auto cluster strategy
            strat = ("column" if self.call_settings["cluster_col"] in dataframe_param.columns
                     else "single") if self.call_settings["cluster_strategy"] is None \
                     else self.call_settings["cluster_strategy"]

            labels = labeler.label_texts(
                dataframe_param,
                provider         = self.call_settings["provider"],
                model_name       = self.call_settings["model_name"],
                openai_api_key   = self.call_settings["openai_api_key"],
                cluster_strategy = strat,
                cluster_col      = self.call_settings["cluster_col"],
                top_n_words      = self.call_settings["top_n_words"],
                num_candidates   = self.call_settings["num_candidates"],
                use_gpu          = self.call_settings["use_gpu"],
            )
            out_csv = base_out_dir / "labels.csv"
            pd.DataFrame(sorted(labels.items()), columns=["cluster", "label"]).to_csv(out_csv, index=False)
            result_dict["<in-memory-df>"] = labels
            written_csvs.append(str(out_csv))

        # register outputs
        bundle[f"{self.tag}.{self.provides[0]}"] = result_dict       # python dict
        bundle[f"{self.tag}.{self.provides[1]}"]  = written_csvs      # list of paths
