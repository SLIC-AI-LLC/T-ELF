from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Sequence, List, Optional, Union
import pandas as pd
import ast

from .base_block import AnimalBlock
from .data_bundle import DataBundle, SAVE_DIR_BUNDLE_KEY


def _nested_dict_splitter(s: Any) -> Dict[str, List[Any]]:
    """
    Generic splitter for JSON-like strings representing nested dictionaries.

    Returns a dict mapping:
      - '' (empty string) -> list of top-level keys
      - each nested key -> flattened list of all values under that key
    """
    try:
        d = ast.literal_eval(s) if isinstance(s, str) else {}
    except Exception:
        return {}

    out: Dict[str, List[Any]] = {'': []}
    for top_key, val in d.items():
        # record the top-level key
        out[''].append(top_key)

        # if the value is a dict, record its nested items
        if isinstance(val, dict):
            for subk, subv in val.items():
                out.setdefault(subk, [])
                if isinstance(subv, (list, tuple)):
                    out[subk].extend(subv)
                else:
                    out[subk].append(subv)
        # if the value is a list, record items under ''
        elif isinstance(val, (list, tuple)):
            out[''].extend(val)
        else:
            # record simple scalar values under their field name
            out.setdefault(top_key, []).append(val)
    return out


def _default_splitter(s: Any) -> List[Any]:
    """
    Default splitter: semicolon-split for strings, wrap others in a single-item list.
    """
    if isinstance(s, str):
        return s.split(";")
    return [s]


class UniquenessDFBlock(AnimalBlock):
    """
    Summarize uniqueness of each column in a DataFrame.

    Needs: 'df'
    Provides: 'uniqueness_df'

    Parameters
    ----------
    splitters : Optional[Dict[str, Callable[[Any], Union[Sequence[Any], Dict[str, List[Any]]]]]]
        Column-specific split functions. Overrides defaults.
        A splitter may return a flat sequence or a dict of sub-sequences.
    exclude_cols : Sequence[str], optional
        Column names to skip entirely (defaults to ['title','abstract','num_citations','num_references']).
    """
    DEFAULT_SPLITTERS: Dict[str, Callable[[Any], Any]] = {
        # semicolon-delimited lists
        "authors":       lambda s: s.split(";") if isinstance(s, str) else [],
        "author_ids":    lambda s: s.split(";") if isinstance(s, str) else [],
        "s2_authors":    lambda s: s.split(";") if isinstance(s, str) else [],
        "s2_author_ids": lambda s: s.split(";") if isinstance(s, str) else [],
        "PACs":          lambda s: s.split(";") if isinstance(s, str) else [],
        "subject_areas": lambda s: s.split(";") if isinstance(s, str) else [],
        "citations":     lambda s: s.split(";") if isinstance(s, str) else [],
        "references":    lambda s: s.split(";") if isinstance(s, str) else [],
        # JSON-like dicts: use generic nested splitter
        "affiliations":  _nested_dict_splitter,
        "funding":       _nested_dict_splitter,
    }
    CANONICAL_NEEDS = ('df',)

    def __init__(
        self,
        splitters: Optional[Dict[str, Callable[[Any], Any]]] = None,
        exclude_cols: Sequence[str] = ('title', 'abstract', 'num_citations', 'num_references'),
        needs=CANONICAL_NEEDS,
        provides: Sequence[str] = ('uniqueness_df',),
        **kwargs: Any,
    ) -> None:
        super().__init__(needs=needs, provides=provides, **kwargs)

        self.splitters: Dict[str, Callable[[Any], Any]] = {
            **self.DEFAULT_SPLITTERS
        }
        if splitters:
            self.splitters.update(splitters)

        self.exclude_cols = set(exclude_cols)

    def run(self, bundle: DataBundle) -> DataBundle:
        df: pd.DataFrame = bundle['df']
        result_rows: List[Dict[str, Any]] = []

        for col in df.columns:
            if col in self.exclude_cols:
                continue
            series = df[col].dropna()

            splitter = self.splitters.get(col, _default_splitter)

            # Prepare a dict of sets to collect unique items
            col_sets: Dict[str, set] = {}

            for cell in series:
                try:
                    out = splitter(cell)
                except Exception:
                    out = []

                if isinstance(out, dict):
                    for subkey, items in out.items():
                        target = subkey or ''
                        col_sets.setdefault(target, set())
                        for item in items:
                            if pd.isna(item):
                                continue
                            col_sets[target].add(item)
                else:
                    col_sets.setdefault('', set())
                    for item in out:
                        if pd.isna(item):
                            continue
                        col_sets[''].add(item)

            # Build row entries for each subkey
            for subkey, items in col_sets.items():
                subcol = f"{col}.{subkey}" if subkey else col
                count = len(items)
                # values_str = ';'.join(map(str, sorted(items)))
                # sort heterogenous items by their string representation
                sorted_items = sorted(items, key=lambda x: str(x))
                values_str    = ';'.join(map(str, sorted_items))
                result_rows.append({'col': subcol, 'count': count, 'values': values_str})

        uniqueness_df = pd.DataFrame(result_rows, columns=[ 'count', 'col','values'])


        if SAVE_DIR_BUNDLE_KEY in bundle:
            out_dir = Path(bundle[SAVE_DIR_BUNDLE_KEY]) / self.tag
            out_dir.mkdir(exist_ok=True, parents=True)
            final_csv = out_dir /  f"{self.tag}.csv"
            uniqueness_df.to_csv(final_csv, index=False, encoding="utf-8-sig")
            self.register_checkpoint(self.provides[0], final_csv)

        bundle[f"{self.tag}.{self.provides[0]}"] = uniqueness_df

        return bundle
