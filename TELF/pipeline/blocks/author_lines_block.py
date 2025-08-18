# blocks/expand_and_assign_block.py

from __future__ import annotations
from pathlib import Path
from typing import Dict, Sequence, Any
import pandas as pd
import ast

from .base_block import AnimalBlock
from .data_bundle import DataBundle, SAVE_DIR_BUNDLE_KEY

class ExpandAuthorAffiliationBlock(AnimalBlock):
    """
    needs    : ('df',)
    provides : ('df',) – expanded DataFrame with *_component columns
    tag      : 'ExpandAssign'
    """
    CANONICAL_NEEDS    = ('df',)

    # columns to drop after expansion
    _TO_DROP = [
        'author_ids','authors','affiliations',
        'countries','query','affiliation_names',
        'slic_affiliations','slic_author_ids','type',
        'affiliation_ids','slic_authors','s2_authors',
        's2_author_ids',
    ]

    def __init__(self, tag='ExpandAssign', **kw) -> None:
        super().__init__(
            needs=self.CANONICAL_NEEDS,
            provides=('author_df',),
            tag=tag,
            **kw,
        )

    def run(self, bundle: DataBundle) -> None:
        # ── 1) LOAD & EXPAND ────────────────────────────────────
        src = bundle[self.needs[0]]
        if isinstance(src, (str, Path)):
            df0 = pd.read_csv(src, dtype=str)
        else:
            df0 = src.copy().astype(str)

        # drop rows missing both IDs
        if {'author_ids','authors'}.issubset(df0.columns):
            df0 = df0.dropna(subset=['author_ids','authors'], how='all')

        records: list[Dict[str,Any]] = []
        for _, row in df0.iterrows():
            base = row.drop(['author_ids','authors','affiliations'], errors='ignore').to_dict()

            # build id→name map
            ids   = [i.strip() for i in row['author_ids'].split(';') if i.strip()]
            names = [n.strip() for n in row['authors'].split(';')]
            auth_map = dict(zip(ids, names))

            # parse affiliations JSON if present
            raw_aff = row.get('affiliations')
            try:
                affs = ast.literal_eval(raw_aff) if pd.notna(raw_aff) else {}
            except Exception:
                affs = {}

            if isinstance(affs, dict) and affs:
                # one row per (affiliation × author)
                for aff_id, aff in affs.items():
                    if not isinstance(aff, dict): 
                        continue
                    for aid in aff.get('authors', []):
                        aid = aid.strip()
                        records.append({
                            **base,
                            'author_id':      aid,
                            'author':         auth_map.get(aid, ''),
                            'affiliation_id': aff_id,
                            'affiliation':    aff.get('name'),
                            'country':        aff.get('country'),
                        })
            else:
                # no affiliations → one row per author
                for aid, aname in auth_map.items():
                    records.append({ **base, 'author_id': aid, 'author': aname })

        expanded_df = pd.DataFrame(records)

        # drop everything in one call
        drop_cols = [c for c in self._TO_DROP if c in expanded_df.columns]
        if drop_cols:
            expanded_df = expanded_df.drop(columns=drop_cols)

        # ── 2) AUTO-DISCOVER & ASSIGN COMPONENTS ─────────────────
        root = Path(bundle[SAVE_DIR_BUNDLE_KEY])
        # map of “column_name” → { eid: component_id, … }
        comp_maps: Dict[str, Dict[str,str]] = {}

        # find every CSV under any Wolf* folder
        for csv_path in root.rglob("*component*documents*.csv"):
            rel = csv_path.relative_to(root).parts
            # expect: ( WolfTag, category, comp_id, filename )
            if len(rel) < 3 or not rel[1] or not rel[2].isdigit():
                continue

            category, comp_id = rel[1], rel[2]
            col = f"{category.replace('-', '')}_component"
            comp_dict = comp_maps.setdefault(col, {})

            # read only eid column
            tmp = pd.read_csv(csv_path, usecols=['eid'], dtype=str)
            for eid in tmp['eid'].dropna().astype(str):
                comp_dict[eid] = comp_id

        # now vectorized‐map each column
        for col, mapping in comp_maps.items():
            expanded_df[col] = expanded_df['eid'].astype(str).map(mapping).astype(pd.StringDtype())

        # ── 3) SAVE & CHECKPOINT ───────────────────────────────
        if SAVE_DIR_BUNDLE_KEY in bundle:
            out_dir = Path(bundle[SAVE_DIR_BUNDLE_KEY]) / self.tag
            out_dir.mkdir(parents=True, exist_ok=True)
            out_csv = out_dir / "expanded_with_components.csv"
            expanded_df.to_csv(out_csv, index=False, encoding="utf-8-sig")
            self.register_checkpoint(self.provides[0], out_csv)

        # ── 4) PUSH BACK INTO BUNDLE ────────────────────────────
        bundle[f"{self.tag}.{self.provides[0]}"] = expanded_df
