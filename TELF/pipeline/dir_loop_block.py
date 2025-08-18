from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from .blocks.base_block import AnimalBlock
from .blocks.data_bundle import (
    DataBundle,
    SAVE_DIR_BUNDLE_KEY,
    SOURCE_DIR_BUNDLE_KEY,
    DIR_LIST_BUNDLE_KEY,
    RESULTS_DEFAULT,
)
from .block_manager import BlockManager

import traceback


def _clone_databundle(bundle: DataBundle) -> DataBundle:
    """
    Clone a DataBundle by copying its internal _store dict,
    which includes each base-key bucket and its '_latest' pointer.
    """
    clone = DataBundle()
    clone.__dict__['_store'] = {
        base: bucket.copy() for base, bucket in bundle._store.items()
    }
    return clone


class DirectoryLoopBlock(AnimalBlock):
    """
    For each path in bundle[DIR_LIST_BUNDLE_KEY]:
      - clone the incoming DataBundle via _clone_databundle
      - set SOURCE_DIR_BUNDLE_KEY and 'dir'
      - optionally override SAVE_DIR_BUNDLE_KEY to SOURCE_DIR_BUNDLE_KEY/RESULTS_DEFAULT
      - run subblocks in parallel, isolating logs per run
    Collects each sub-bundle's .as_dict() under bundle[tag.RESULTS_DEFAULT].
    """
    CANONICAL_NEEDS = (DIR_LIST_BUNDLE_KEY,)

    def __init__(
        self,
        subblocks: List[AnimalBlock],
        needs: Tuple[str, ...] = CANONICAL_NEEDS,
        provides: Tuple[str, ...] = (RESULTS_DEFAULT,),
        tag: str = 'DirLoop',
        conditional_needs: Sequence[Tuple[str, Any]] = (),
        capture_output: str = 'file',
        use_source_dir: bool = True,
        *,
        init_settings: Dict[str, Any] | None = None,
        call_settings: Dict[str, Any] | None = None,
        max_workers: int | None = None,
        verbose: bool = True,
        force_checkpoint: bool | None = None,
        **kwargs,
    ) -> None:
        self.use_source_dir = use_source_dir
        self.force_checkpoint = force_checkpoint
        super().__init__(
            needs=needs,
            provides=provides,
            init_settings=self._merge({}, init_settings),
            call_settings=self._merge({}, call_settings),
            conditional_needs=conditional_needs,
            tag=tag,
            verbose=verbose,
            **kwargs,
        )
        self.capture_output = capture_output
        self.subblocks = subblocks
        self.max_workers = max_workers

    def run(self, bundle: DataBundle) -> DataBundle:
        dirs = list(bundle[self.needs[0]])
        total = len(dirs)
        results: List[Dict[str, Any]] = []
        base_bundle = bundle

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_path = {
                executor.submit(self._process_one, base_bundle, path): path
                for path in dirs
            }

            completed = 0
            for future in as_completed(future_to_path):
                path = future_to_path[future]
                try:
                    sub_res = future.result()
                except Exception:
                    if self.verbose:
                        print(f"❌ {path!r} raised an exception:")
                        traceback.print_exc()
                else:
                    results.append(sub_res)
                    completed += 1
                    if self.verbose:
                        print(f"✅ [{completed}/{total}] Completed processing {path!r}")

        bundle[f"{self.tag}.{self.provides[0]}"] = results
        return bundle

    def _process_one(self, base_bundle: DataBundle, path: Path) -> Dict[str, Any]:
        try:
            sub = _clone_databundle(base_bundle)
            sub[SOURCE_DIR_BUNDLE_KEY] = Path(path)
            sub['dir'] = Path(path)

            if self.use_source_dir:
                sub[SAVE_DIR_BUNDLE_KEY] = sub[SOURCE_DIR_BUNDLE_KEY] / RESULTS_DEFAULT

            if self.verbose:
                print(f"\n▶  Starting {path!r}")
                print(f"    Output directory: {sub[SAVE_DIR_BUNDLE_KEY]!r}")

            mgr = BlockManager(
                blocks=self.subblocks,
                databundle=sub,
                verbose=self.verbose,
                capture_output=self.capture_output,
                force_checkpoint=self.force_checkpoint,
            )
            result_bundle = mgr()
            return result_bundle.as_dict()

        except Exception:
            print(f"⚠️ Error processing {path!r} in _process_one:")
            traceback.print_exc()
            raise
