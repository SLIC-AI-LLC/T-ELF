from pathlib import Path
from typing import Dict, Sequence, Any
import pandas as pd
import nest_asyncio
nest_asyncio.apply()
from pymongo import ASCENDING

from .base_block import AnimalBlock
from .data_bundle import DataBundle, SAVE_DIR_BUNDLE_KEY
from ...pre_processing import Scopus
from ...applications.Penguin import Penguin

from ...helpers.file_system import copy_all_files

class ScopusBlock(AnimalBlock):
    CANONICAL_NEEDS = ('df', )

    def __init__(
        self,
        use_penguin: bool = False,
        penguin_settings: Dict[str, Any] = {
            "uri": "localhost:27017",
            "db_name": "Penguin",
            "username": None,
            "password": None,
        },
        needs: Sequence[str] = CANONICAL_NEEDS,
        provides: Sequence[str] = ("df",),
        tag: str = "Scopus",
        conditional_needs: Sequence[tuple[str, Any]] = (),
        *,
        init_settings: Dict[str, Any] = None,
        call_settings: Dict[str, Any] = None,
        **kw,
    ) -> None:
        self.use_penguin = use_penguin
        self.penguin_settings = penguin_SETTINGS = penguin_settings

        default_init = {
            'keys': None,
            'mode': 'fs',
            'name': None,      # real cache dir if provided
            'ignore': None,
            'verbose': True,
        }
        super().__init__(
            needs=needs,
            provides=provides,
            init_settings=self._merge(default_init, init_settings),
            call_settings={},
            conditional_needs=conditional_needs,
            tag=tag,
            **kw,
        )

    def run(self, bundle: DataBundle) -> None:
        # 1) Load input & extract DOIs
        raw = bundle[self.needs[0]]
        df_in = self.load_path(raw) if isinstance(raw, (str, Path)) else raw.copy()
        dois = df_in.doi.dropna().astype(str).tolist()
        if self.verbose:
            print(f"[{self.tag}] candidate DOIs count: {len(dois)}")

        # 2) Prepare output dirs & CSV
        out_dir = Path(bundle[SAVE_DIR_BUNDLE_KEY]) / self.tag
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / 'scopus.csv'
        cache_dir = out_dir / 'SCOPUS_CACHE'
        cache_dir.mkdir(exist_ok=True, parents=True)
        if self.verbose:
            print(f"[{self.tag}] cache dir: {cache_dir}")

        # 3) Connect to Penguin if requested
        if self.use_penguin:
            penguin = Penguin(**self.penguin_settings)
            doi_field = penguin.scopus_attributes['doi']
            col = penguin.db[Penguin.SCOPUS_COL]
            col.create_index([(doi_field, ASCENDING)])
            seen_db = set(col.distinct(doi_field, {doi_field: {'$in': dois}}))
        else:
            seen_db = set()
        if self.verbose:
            print(f"[{self.tag}] seen in Mongo: {len(seen_db)}")

        # 4) Load or initialize local CSV
        if csv_path.exists():
            df_csv = pd.read_csv(csv_path)
        else:
            df_csv = pd.DataFrame(columns=df_in.columns)
        seen_local = set(df_csv.doi.dropna().astype(str).tolist())
        if self.verbose:
            print(f"[{self.tag}] seen locally: {len(seen_local)}")

        # 5) Figure out what to fetch from Scopus API
        to_fetch = [d for d in dois if d not in seen_local and d not in seen_db]
        if self.verbose:
            print(f"[{self.tag}] to fetch count: {len(to_fetch)}")

        # 6) Fetch any new records
        df_new = pd.DataFrame(columns=df_csv.columns)
        if to_fetch:
            settings = dict(self.init_settings, name=str(cache_dir))
            sc = Scopus(**settings)
            query = ' OR '.join(f"DOI({doi})" for doi in to_fetch)
            try:
                df_new, _ = sc.search(query)
            except ValueError as e:
                if "No Scopus papers" in str(e):
                    df_new = pd.DataFrame(columns=df_csv.columns)
                else:
                    raise
            if self.verbose:
                print(f"[{self.tag}] fetched via API: {len(df_new)}")

            # 6a) upsert those JSONs into Penguin
            if self.use_penguin and not df_new.empty:
                penguin.add_many_documents(cache_dir, source='Scopus', overwrite=True)
                if self.verbose:
                    print(f"[{self.tag}] bulk-upserted JSON files into Mongo")

                # 6b) re-load only the ones we just added
                df_id = penguin.id_search(
                    ids=[f"doi:{doi}" for doi in to_fetch],
                    as_pandas=True
                )
                id_key = penguin.scopus_attributes['id']
                df_new = df_id[df_id[id_key].notnull()].reset_index(drop=True)
            if self.verbose:
                print(f"[{self.tag}] entries from DB after upsert: {len(df_new)}")

        # 7) Pull any DB-only records we didn’t fetch locally
        df_db = pd.DataFrame(columns=df_csv.columns)
        if self.use_penguin:
            missing_db = seen_db - seen_local
            if missing_db:
                docs = list(col.find({doi_field: {'$in': list(missing_db)}}))
                df_db = pd.DataFrame(docs).drop(columns=['_id'], errors='ignore')
                if self.verbose:
                    print(f"[{self.tag}] added DB-only records: {len(df_db)}")

        # 8) Merge, dedupe, save, move cache, checkpoint
        df_all = pd.concat([df_csv, df_new, df_db], ignore_index=True)
        df_all = df_all.drop_duplicates(subset=['doi']).reset_index(drop=True)
        df_all.to_csv(csv_path, index=False)
        if self.verbose:
            print(f"[{self.tag}] total after merge: {len(df_all)}")

        # 9) Optionally relocate the flat‐file cache into a “real” cache dir
        real_cache = self.init_settings.get('name')
        if real_cache:
            copied = copy_all_files(cache_dir, real_cache, preserve_metadata=True)
            if self.verbose:
                print(f"[{self.tag}] copied {copied} files to real cache at {real_cache}")


        self.register_checkpoint(self.provides[0], str(csv_path))
        bundle[f"{self.tag}.{self.provides[0]}"] = df_all
