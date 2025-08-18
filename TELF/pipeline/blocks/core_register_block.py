# blocks/merge_scopus_s2_block.py
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple
from .base_block import AnimalBlock
from .data_bundle import DataBundle, SAVE_DIR_BUNDLE_KEY

class CoreRegister(AnimalBlock):
    CANONICAL_NEEDS = ('df', )

    def __init__(
        self,
        *,
        needs: Sequence[str] =  CANONICAL_NEEDS,
        provides: Sequence[str] = ("df",),
        tag: str = "CoreDf",
        init_settings: Dict[str, Any] | None = None,
        call_settings: Dict[str, Any] | None = None,
        verbose: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            needs=needs,
            provides=provides,
            tag=tag,
            init_settings=self._merge({"verbose": True}, init_settings),
            call_settings=self._merge({}, call_settings),
            verbose=verbose,
            **kwargs,
        )

    def run(self, bundle: DataBundle) -> None:
        df = bundle[self.needs[0]]
        if SAVE_DIR_BUNDLE_KEY in bundle:
            out_dir = Path(bundle[SAVE_DIR_BUNDLE_KEY]) / self.tag
            out_dir.mkdir(exist_ok=True, parents=True)
            final_csv = out_dir / "core.csv"
            df.to_csv(final_csv, index=False, encoding="utf-8-sig")
            self.register_checkpoint(self.provides[0], final_csv)
        bundle[f"{self.tag}.{self.provides[0]}"] = df
