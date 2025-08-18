from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
import traceback

from .blocks.base_block import AnimalBlock
from .blocks.data_bundle import DataBundle, SAVE_DIR_BUNDLE_KEY, RESULTS_DEFAULT
from .block_manager import BlockManager


def _clone_databundle(bundle: DataBundle) -> DataBundle:
    """
    Shallow-clone a DataBundle, preserving each base-key bucket and its
    '_latest' pointer.
    """
    clone = DataBundle()
    clone.__dict__['_store'] = {
        base: bucket.copy() for base, bucket in bundle._store.items()
    }
    return clone


class RepeatLoopBlock(AnimalBlock):
    """
    Run a list of `subblocks` `n_iter` times.
    Its .needs are taken from the *first* subblock, and its .provides is only RESULTS_DEFAULT.

    Ensures all iterations write into a single `<SAVE_DIR>/RepeatLoop/iter_xx` tree,
    without nesting off of previous iterations. After completion, restores
    the original SAVE_DIR so subsequent blocks write to the correct location.

    If `stop_on_error` is True (default), the loop aborts on the first iteration error,
    returning only the successful results so far.
    """

    def __init__(
        self,
        subblocks: List[AnimalBlock],
        *,
        n_iter: int,
        clone: bool = False,
        parallel: bool = False,
        max_workers: int | None = None,
        redirect_save_dir: bool = True,
        tag: str = "RepeatLoop",
        capture_output: str = "file",
        verbose: bool = True,
        stop_on_error: bool = True,
        force_checkpoint: bool | None = None,
        **kwargs: Any,
    ) -> None:
        self.subblocks = subblocks
        self.n_iter = int(n_iter)
        self.clone = clone
        self.parallel = parallel
        self.max_workers = max_workers
        self.redirect_save_dir = redirect_save_dir
        self.capture_output = capture_output
        self.verbose = verbose
        self.stop_on_error = stop_on_error
        self.force_checkpoint = force_checkpoint

        # take needs from the first subblock only:
        first = subblocks[0]
        needs = tuple(first.needs) + tuple(key for key, _ in first.conditional_needs)

        # this block itself only provides the collected results list
        provides = (RESULTS_DEFAULT,)

        super().__init__(
            needs=needs,
            provides=provides,
            tag=tag,
            verbose=verbose,
            capture_output=capture_output,
            **kwargs,
        )

    def run(self, bundle: DataBundle) -> DataBundle:
        results: List[Dict[str, Any]] = []
        total = self.n_iter

        # Preserve the original save directory
        orig_save = bundle.get(SAVE_DIR_BUNDLE_KEY)

        # Determine base directory for all iterations
        base_save = Path(orig_save) if orig_save is not None else Path.cwd() / "results"
        loop_root = base_save / self.tag if self.redirect_save_dir else None

        if self.parallel:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {
                    executor.submit(self._one_pass, bundle, i, loop_root): i
                    for i in range(total)
                }
                completed = 0
                for fut in as_completed(futures):
                    i = futures[fut]
                    try:
                        res = fut.result()
                    except Exception:
                        if self.verbose:
                            print(f"❌ iter={i} raised exception:")
                            traceback.print_exc()
                        if self.stop_on_error:
                            for pending in futures:
                                if not pending.done():
                                    pending.cancel()
                            break
                        else:
                            continue
                    else:
                        results.append(res)
                        completed += 1
                        if self.verbose:
                            print(f"✅ [{completed}/{total}] finished iter={i}")
        else:
            for i in range(total):
                if self.verbose:
                    print(f"\n▶  [{i+1}/{total}] starting iter={i}")
                try:
                    res = self._one_pass(bundle, i, loop_root)
                except Exception:
                    if self.stop_on_error:
                        if self.verbose:
                            print(f"⏹ stopping after iter={i} due to error.")
                        break
                    else:
                        continue
                else:
                    results.append(res)
                    if self.verbose:
                        print(f"✓  [{i+1}/{total}] finished iter={i}")

        # store whatever successful results we have
        bundle[f"{self.tag}.{RESULTS_DEFAULT}"] = results

        # Restore original save directory for subsequent blocks
        if orig_save is not None:
            bundle[SAVE_DIR_BUNDLE_KEY] = orig_save

        return bundle

    def _one_pass(self, base_bundle: DataBundle, idx: int, loop_root: Path | None) -> Dict[str, Any]:
        try:
            sub = _clone_databundle(base_bundle) if self.clone else base_bundle
            sub['iter'] = idx

            if loop_root is not None:
                sub[SAVE_DIR_BUNDLE_KEY] = loop_root / f"iter_{idx:02d}"

            mgr = BlockManager(
                blocks=self.subblocks,
                databundle=sub,
                verbose=self.verbose,
                capture_output=self.capture_output,
                force_checkpoint=self.force_checkpoint,
            )
            return mgr().as_dict()

        except Exception:
            print(f"⚠️ Error in RepeatLoopBlock (iter={idx}):")
            traceback.print_exc()
            raise
