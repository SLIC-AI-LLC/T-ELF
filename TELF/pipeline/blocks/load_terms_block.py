from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union
from collections import defaultdict

from .base_block import AnimalBlock
from .data_bundle import DataBundle, SOURCE_DIR_BUNDLE_KEY
from ...applications import CheetahTermFormatter
from ...helpers.terms import resolve_substitution_conflicts

def find_markdown_file(
    dir_path: Optional[Path],
    *,
    explicit: Optional[Path] = None,
    preferred_name: str = "single_stage_terms.md",
) -> Path:
    if explicit is None and dir_path is None:
        raise FileNotFoundError("No markdown file specified and SOURCE_DIR is missing.")

    if explicit:
        md = Path(explicit)
        if not md.exists():
            raise FileNotFoundError(f"Explicit markdown path {md} does not exist.")
        return md

    candidates = list(dir_path.glob("*.md"))
    if not candidates:
        raise FileNotFoundError(f"No markdown files found in {dir_path}")

    for c in candidates:
        if c.name == preferred_name:
            return c
    return candidates[0]

class LoadTermsBlock(AnimalBlock):
    """
    Load a Markdown term list (or passed-in overrides) and place four
    artefacts into the bundle:

        • <tag>.terms                  (raw terms list)
        • <tag>.substitutions          (forward map)
        • <tag>.substitutions_reverse  (reverse map)
        • <tag>.query                  (Cheetah-ready query list)

    If drop_conflicts is True, mappings are conflict-resolved.
    """

    CANONICAL_NEEDS: Sequence[str] = ()

    def __init__(
        self,
        *,
        needs: Sequence[str] = CANONICAL_NEEDS,
        provides: Sequence[str] = ("terms", "substitutions", "substitutions_reverse", "query"),
        tag: str = "Terms",
        init_settings: Optional[Dict[str, Any]] = None,
        call_settings: Optional[Dict[str, Any]] = None,
    ):
        default_call = {
            SOURCE_DIR_BUNDLE_KEY: None,
            "overrides": None,
            "drop_conflicts": True,
        }
        merged_call = self._merge(default_call, call_settings)

        conds = [
            (
                SOURCE_DIR_BUNDLE_KEY,
                lambda bundle, blk: blk.call_settings[SOURCE_DIR_BUNDLE_KEY] is None,
            )
        ]

        super().__init__(
            needs=needs,
            provides=provides,
            tag=tag,
            conditional_needs=conds,
            init_settings=self._merge({}, init_settings),
            call_settings=merged_call,
        )
        self.tag = tag

    def run(self, bundle: DataBundle) -> None:
        overrides = self.call_settings["overrides"]
        drop_conflicts = self.call_settings["drop_conflicts"]

        if overrides is not None:
            sub_forward, sub_reverse, terms = overrides
        else:
            dir_path = (
                Path(bundle[SOURCE_DIR_BUNDLE_KEY])
                if SOURCE_DIR_BUNDLE_KEY in bundle
                else None
            )
            explicit = self.call_settings[SOURCE_DIR_BUNDLE_KEY]
            md_file = find_markdown_file(
                dir_path,
                explicit=Path(explicit) if explicit else None,
            )

            fmt = CheetahTermFormatter(
                markdown_file=md_file,
                lower=True,
                substitutions=True,
                drop_conflicts=drop_conflicts,
            )
            sub_forward, sub_reverse = fmt.get_substitution_maps()
            terms = fmt.get_terms()

        if drop_conflicts:
            clean_forward, dropped = resolve_substitution_conflicts(sub_forward, warn=True)
            if dropped:
                terms = self._prune_terms_list(terms, dropped)
        else:
            clean_forward = sub_forward

        # build reverse map
        rev: Dict[str, List[str]] = defaultdict(list)
        for src, tgt in clean_forward.items():
            rev[tgt].append(src)
        clean_reverse = dict(rev)

        # stash raw terms + maps
        bundle[f"{self.tag}.terms"] = terms
        bundle[f"{self.tag}.substitutions"] = clean_forward
        bundle[f"{self.tag}.substitutions_reverse"] = clean_reverse

        # build Cheetah-ready query
        def _flatten(entry: Any) -> Union[str, Dict[str, List[str]]]:
            if isinstance(entry, str):
                return entry
            if isinstance(entry, dict) and len(entry) == 1:
                term, spec = next(iter(entry.items()))
                # nested positives/negatives
                if isinstance(spec, dict) and 'positives' in spec and 'negatives' in spec:
                    pos = [f"+{p}" for p in spec['positives']]
                    neg = spec['negatives']
                    return {term: pos + neg}
                # direct list of dependents
                if isinstance(spec, list):
                    return {term: spec}
            raise TypeError(f"Cannot normalize term entry: {entry!r}")

        query_list: List[Any] = [_flatten(e) for e in terms]
        bundle[f"{self.tag}.query"] = query_list

    @staticmethod
    def _prune_terms_list(terms: List[Any], dropped: set[str]) -> List[Any]:
        pruned: List[Any] = []
        for entry in terms:
            if isinstance(entry, str):
                if entry not in dropped:
                    pruned.append(entry)
            else:
                kept = {k: v for k, v in entry.items() if k not in dropped}
                if kept:
                    pruned.append(kept)
        return pruned
