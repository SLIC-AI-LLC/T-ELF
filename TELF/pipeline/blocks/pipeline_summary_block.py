from __future__ import annotations
from pathlib import Path
from typing import Dict, Sequence, Any, Tuple, List
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt

from .base_block import AnimalBlock
from .data_bundle import DataBundle, SAVE_DIR_BUNDLE_KEY


class PipelineSummaryBlock(AnimalBlock):
    """
    Final pipeline block: scan every block directory, pick up each
    block's 'df' checkpoint (if present), count its rows, and plot
    them in execution order.
    """

    def __init__(
        self,
        *,
        needs: Sequence[str] = (),
        provides: Sequence[str] = ("docs_summary_df", "docs_summary_plot"),
        tag: str = "PipelineSummary",
        checkpoint_keys: Sequence[str] = ("docs_summary_df", "docs_summary_plot"),
        **kw,
    ):
        super().__init__(
            needs=needs,
            provides=provides,
            tag=tag,
            checkpoint_keys=checkpoint_keys,
            **kw,
        )

    def _discover(self, root: Path) -> List[Tuple[str, Path, float]]:
        """
        Return a list of (block_tag, df_file, mtime) for each sub-directory
        that has a __checkpoints__.json with a 'df' entry.
        """
        found: List[Tuple[str, Path, float]] = []

        for blk_dir in root.iterdir():
            if not blk_dir.is_dir():
                continue
            tag = blk_dir.name
            ckpt = blk_dir / "__checkpoints__.json"
            if not ckpt.is_file():
                continue

            try:
                mapping = json.loads(ckpt.read_text(encoding="utf-8"))
            except Exception:
                continue

            df_path_str = mapping.get("df")
            if not df_path_str:
                continue

            fp = Path(df_path_str)
            if not fp.is_file():
                continue

            mtime = fp.stat().st_mtime
            found.append((tag, fp, mtime))

        found.sort(key=lambda tup: tup[2])
        return found

    def run(self, bundle: DataBundle) -> None:
        if SAVE_DIR_BUNDLE_KEY not in bundle:
            raise KeyError(
                f"{self.tag}: bundle lacks {SAVE_DIR_BUNDLE_KEY}; cannot discover checkpoint folders."
            )

        root = Path(bundle[SAVE_DIR_BUNDLE_KEY])
        triples = self._discover(root)

        rows: List[Tuple[str, int, str]] = []
        for tag, fp, mtime in triples:
            try:
                obj = self.load_path(fp)
                count = obj.shape[0] if hasattr(obj, "shape") else len(obj)
            except Exception:
                count = -1
            rows.append((
                tag,
                count,
                dt.datetime.fromtimestamp(mtime).isoformat(sep=" ", timespec="seconds")
            ))

        summary_df = pd.DataFrame(
            rows,
            columns=["block_tag", "num_docs", "first_write_time"]
        )

        # save CSV
        out_dir = root / self.tag
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / "docs_per_block.csv"
        img_path = out_dir / "docs_per_block.png"
        summary_df.to_csv(csv_path, index=False)

        # plotting with log scale, individual colors, and annotations
        n_bars = len(summary_df)
        cmap = plt.get_cmap("tab20")
        colors = cmap(np.linspace(0, 1, n_bars))

        plt.figure(figsize=(max(4, 0.5 * n_bars), 4))
        bars = plt.bar(
            summary_df["block_tag"],
            summary_df["num_docs"],
            color=colors,
        )
        plt.yscale("log")

        for bar, count in zip(bars, summary_df["num_docs"]):
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{count}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        plt.xticks(rotation=45, ha="right")
        plt.ylabel("Documents")
        plt.title("Documents per Block (execution order)")
        plt.tight_layout()
        plt.savefig(img_path, dpi=150)
        plt.close()

        # bundle values + checkpoints
        bundle[f"{self.tag}.docs_summary_df"] = summary_df
        bundle[f"{self.tag}.docs_summary_plot"] = img_path
        self.register_checkpoint("docs_summary_df", csv_path)
        self.register_checkpoint("docs_summary_plot", img_path)
