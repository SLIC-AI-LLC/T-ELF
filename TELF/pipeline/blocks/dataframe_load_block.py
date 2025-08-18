from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence, Tuple, Union
import re
import pandas as pd

from .base_block import AnimalBlock
from .data_bundle import DataBundle, SOURCE_DIR_BUNDLE_KEY, RESULTS_DEFAULT

class LoadDfBlock(AnimalBlock):
    """
    Load CSV file(s) under <bundle['dir']>/<path_extension>, or directly from a full path.

    - If `full_path` is set, ignores the bundle and path_extension and loads from that path.
    - Else if `recursive=True` or `regex` is provided, searches under the subfolder,
      filters by regex, and returns either a single DataFrame or a list of file paths.
    - Otherwise, loads exactly the file at <bundle['dir']>/<path_extension>.
    """
    CANONICAL_NEEDS = (SOURCE_DIR_BUNDLE_KEY,)

    def __init__(
        self,
        *,
        needs: Sequence[str] = CANONICAL_NEEDS,
        provides: Sequence[str] = ("df", "df_paths"),
        path_extension: Path | str = Path(RESULTS_DEFAULT) / "papers.csv",
        recursive: bool = False,
        regex: str | None = None,
        return_multiple: bool = False,
        full_path: Path | str | None = None,
        conditional_needs: Sequence[Tuple[str, Any]] = (),
        tag: str = "LoadDF",
        verbose: bool = True,
        **kw: Any,
    ) -> None:
        super().__init__(
            needs=needs,
            provides=provides,
            conditional_needs=conditional_needs,
            tag=tag,
            verbose=verbose,
            **kw,
        )
        self.path_extension = Path(path_extension)
        self.recursive = recursive
        self.regex = regex
        self._pattern = re.compile(regex) if regex else None
        self.return_multiple = return_multiple
        self.full_path = Path(full_path) if full_path is not None else None

    def run(self, bundle: DataBundle) -> None:
        # Determine base search location
        if self.full_path:
            search_root = self.full_path
        else:
            base_dir = Path(bundle[self.needs[0]])
            search_root = base_dir / self.path_extension

        # Collect CSV paths
        if search_root.is_file():
            matches = [search_root]
        elif (self.recursive or self.regex) and search_root.is_dir():
            globber = search_root.rglob if self.recursive else search_root.glob
            candidates = globber("*.csv")
            matches = [p for p in candidates if not self._pattern or self._pattern.search(p.name)]
        elif not (self.recursive or self.regex) and search_root.exists():
            matches = [search_root]
        else:
            # No matches or wrong type
            if self.full_path:
                raise FileNotFoundError(f"No CSV files found at specified full_path {search_root!r}")
            raise FileNotFoundError(
                f"No CSV files matched under {search_root!r}"
                + (f" with regex={self.regex!r}" if self.regex else "")
            )

        # Deduplicate and sort
        matches = sorted(set(matches))

        # Load or return paths
        if len(matches) == 1 or not self.return_multiple:
            df = pd.read_csv(matches[0])
            bundle[f"{self.tag}.df"] = df
            bundle[f"{self.tag}.df_paths"] = []
        else:
            bundle[f"{self.tag}.df"] = None
            bundle[f"{self.tag}.df_paths"] = matches
