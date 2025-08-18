from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Sequence, List

import pandas as pd
import matplotlib.pyplot as plt

from .base_block import AnimalBlock
from .data_bundle import DataBundle, SAVE_DIR_BUNDLE_KEY
from .pipeline_summary_block import PipelineSummaryBlock


def _bundle_items(db: DataBundle):
    """
    Iterate over a DataBundle as (pseudo-key, value) pairs so the collectors can
    treat it like a regular mapping.

    This yields:
        - "Tag.docs_summary_df" for docs_summary_df entries
        - "Tag.df" for any *.df entries
        - other values are propagated for recursion

    Parameters
    ----------
    db : DataBundle
        The data bundle to iterate.

    Yields
    ------
    tuple[str, Any]
        Pseudo-key and corresponding value.
    """
    for base, bucket in db._store.items():  # noqa: SLF001
        for tag, val in bucket.items():
            if tag == "_latest":
                continue
            yield f"{tag}.{base}", val


class GroupSummaryBlock(AnimalBlock):
    """
    Aggregate document counts across any mixture of RepeatLoopBlock and
    DirectoryLoopBlock nests. Falls back to counting every `*.df` when no
    PipelineSummaryBlock `docs_summary_df` objects are present.

    Parameters
    ----------
    summary_block_tag : str, optional
        Tag used by PipelineSummaryBlock to generate summary dataframes,
        by default "PipelineSummary"
    call_settings : dict[str, Any] | None, optional
        Settings for plotting and ordering, by default None
    needs : sequence[str] | None, optional
        Keys required from the bundle, by default None (uses SAVE_DIR_BUNDLE_KEY)
    provides : sequence[str], optional
        Keys provided to the bundle after run, by default
        ("group_summary_df", "group_summary_plot")
    tag : str, optional
        Tag for this block, by default "GroupSummary"
    """

    def __init__(
        self,
        *,
        summary_block_tag: str = "PipelineSummary",
        call_settings: Dict[str, Any] | None = None,
        needs: Sequence[str] | None = None,
        provides: Sequence[str] = ("group_summary_df", "group_summary_plot"),
        tag: str = "GroupSummary",
        **kw: Any,
    ):
        default_call = {
            "log_scale": True,
            "plot_kwargs": {},
            "label_bars": True,
            "order_by_pipeline": False,
            "desired_order": []
        }
        if needs is None:
            needs = (SAVE_DIR_BUNDLE_KEY,)
        super().__init__(
            needs=tuple(needs),
            provides=provides,
            tag=tag,
            call_settings=self._merge(default_call, call_settings),
            **kw
        )
        self.summary_tag = summary_block_tag

    def _is_summary_df(self, key: str) -> bool:
        """
        Determine if a key corresponds to a PipelineSummaryBlock dataframe.

        Parameters
        ----------
        key : str
            The bundle key to check.

        Returns
        -------
        bool
            True if it matches docs_summary_df pattern.
        """
        return (
            key == "docs_summary_df"
            or (key.endswith(".docs_summary_df") and key.startswith(self.summary_tag))
        )

    def _ctx_label(self, ctx: Dict[str, str]) -> str:
        """
        Construct a group label from directory and iteration context.

        Parameters
        ----------
        ctx : dict[str, str]
            Context dictionary containing 'dir' and/or 'iter'.

        Returns
        -------
        str
            Combined label, e.g. "dir/iter" or single value.
        """
        if "dir" in ctx and "iter" in ctx:
            return f"{ctx['dir']}/{ctx['iter']}"
        return ctx.get("dir") or ctx.get("iter") or "all"

    def _collect_summaries(
        self, obj: Any, ctx: Dict[str, str], out: List[pd.DataFrame]
    ) -> None:
        """
        Recursively collect PipelineSummaryBlock dataframes and tag with group.

        Parameters
        ----------
        obj : Any
            Nested object (dict, list, DataBundle).
        ctx : dict[str, str]
            Current directory/iteration context.
        out : list[pandas.DataFrame]
            Collected dataframes with added 'group' column.
        """
        if isinstance(obj, dict):
            new_ctx = ctx.copy()
            if "dir" in obj and obj["dir"] is not None:
                new_ctx["dir"] = Path(obj["dir"]).name
            if "iter" in obj and obj["iter"] is not None:
                new_ctx["iter"] = f"iter_{int(obj['iter'])}"
            for k, v in obj.items():
                if self._is_summary_df(k) and isinstance(v, pd.DataFrame):
                    df = v.copy()
                    df["group"] = self._ctx_label(new_ctx)
                    out.append(df)
                if isinstance(v, (dict, list, DataBundle)):
                    self._collect_summaries(v, new_ctx, out)

        elif isinstance(obj, list):
            for item in obj:
                self._collect_summaries(item, ctx, out)

        elif isinstance(obj, DataBundle):
            for k, v in _bundle_items(obj):
                if self._is_summary_df(k) and isinstance(v, pd.DataFrame):
                    df = v.copy()
                    df["group"] = self._ctx_label(ctx)
                    out.append(df)
                if isinstance(v, (dict, list, DataBundle)):
                    self._collect_summaries(v, ctx, out)

    def _collect_raw_dfs(
        self, obj: Any, ctx: Dict[str, str], out: List[pd.DataFrame]
    ) -> None:
        """
        Recursively collect any DataFrame (*.df) and count rows as fallback.

        Parameters
        ----------
        obj : Any
            Nested object to search.
        ctx : dict[str, str]
            Current context.
        out : list[pandas.DataFrame]
            Collected dataframes with block_tag, num_docs, group.
        """
        if isinstance(obj, dict):
            new_ctx = ctx.copy()
            if "dir" in obj and obj["dir"] is not None:
                new_ctx["dir"] = Path(obj["dir"]).name
            if "iter" in obj and obj["iter"] is not None:
                new_ctx["iter"] = f"iter_{int(obj['iter'])}"
            for k, v in obj.items():
                if k.endswith(".df") and isinstance(v, pd.DataFrame):
                    block_tag = k.split(".")[0]
                    out.append(
                        pd.DataFrame({
                            "block_tag": [block_tag],
                            "num_docs": [len(v)],
                            "group": [self._ctx_label(new_ctx)],
                        })
                    )
                if isinstance(v, (dict, list, DataBundle)):
                    self._collect_raw_dfs(v, new_ctx, out)

        elif isinstance(obj, list):
            for item in obj:
                self._collect_raw_dfs(item, ctx, out)

        elif isinstance(obj, DataBundle):
            for k, v in _bundle_items(obj):
                if k.endswith(".df") and isinstance(v, pd.DataFrame):
                    block_tag = k.split(".")[0]
                    out.append(
                        pd.DataFrame({
                            "block_tag": [block_tag],
                            "num_docs": [len(v)],
                            "group": [self._ctx_label(ctx)],
                        })
                    )
                if isinstance(v, (dict, list, DataBundle)):
                    self._collect_raw_dfs(v, ctx, out)

    def run(self, bundle: DataBundle) -> DataBundle:
        """
        Execute the block: collect summaries or raw dfs, pivot, reorder columns,
        and generate/save both CSV and bar plot with optional labels.

        Parameters
        ----------
        bundle : DataBundle
            The input data bundle containing dfs and summary dfs.

        Returns
        -------
        DataBundle
            The same bundle with added:
            - "GroupSummary.group_summary_df" (pandas.DataFrame)
            - "GroupSummary.group_summary_plot" (Path to PNG)

        Raises
        ------
        ValueError
            If no DataFrames are found to summarise.
        """
        # Ensure at least one PipelineSummary exists
        if not any(self._is_summary_df(k) for k in bundle):
            try:
                PipelineSummaryBlock(tag=self.summary_tag)(bundle)
            except Exception:
                pass

        dfs: List[pd.DataFrame] = []
        self._collect_summaries(bundle, {}, dfs)

        if not dfs:
            self._collect_raw_dfs(bundle, {}, dfs)

        if not dfs:
            raise ValueError(f"{self.tag}: could not locate any DataFrames to summarise.")

        combined = pd.concat(dfs, ignore_index=True)
        pivot = (
            combined
            .pivot_table(
                index="group", columns="block_tag",
                values="num_docs", aggfunc="sum", fill_value=0
            )
            .apply(pd.to_numeric, errors="coerce").fillna(0).astype(int)
        )

        # Reorder columns if requested
        if self.call_settings.get("pipeline_order", False):
            pipeline_order = self.call_settings.get("pipeline_order", [])
            if pipeline_order:
                valid = [col for col in pipeline_order if col in pivot.columns]
                pivot = pivot.reindex(columns=valid)

        # Save and plot
        out_dir = Path(bundle[SAVE_DIR_BUNDLE_KEY]) / self.tag
        out_dir.mkdir(parents=True, exist_ok=True)

        csv_path = out_dir / "group_summary.csv"
        pivot.to_csv(csv_path)

        png_path = out_dir / "group_summary.png"
        if not pivot.empty and pivot.values.sum() > 0:
            fig, ax = plt.subplots(
                figsize=(max(6, 0.3 * pivot.shape[1]),
                         max(4, 0.5 * pivot.shape[0]))
            )
            pivot.plot(
                kind="bar", stacked=False, ax=ax,
                **self.call_settings.get("plot_kwargs", {})
            )
            ax.set_xlabel("Group")
            ax.set_ylabel("Documents")
            ax.set_title("Documents per Block by Group")
            # rotate and align x-tick labels
            ax.tick_params(axis="x", labelrotation=45)
            for lbl in ax.get_xticklabels():
                lbl.set_ha('right')

            if self.call_settings.get("label_bars", True):
                for container in ax.containers:
                    for bar in container:
                        height = bar.get_height()
                        if height > 0:
                            ax.text(
                                bar.get_x() + bar.get_width() / 2,
                                height, str(height),
                                ha="center", va="bottom", fontsize=8,
                                rotation=90                         # rotation angle
                            )

            plt.tight_layout()
            if self.call_settings['log_scale']:
                plt.yscale("log")
            plt.savefig(png_path, dpi=150)
            plt.close()

        bundle[f"{self.tag}.group_summary_df"] = pivot
        bundle[f"{self.tag}.group_summary_plot"] = png_path
        self.register_checkpoint("group_summary_df", csv_path)
        self.register_checkpoint("group_summary_plot", png_path)
        return bundle
