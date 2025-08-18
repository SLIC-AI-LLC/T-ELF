from pathlib import Path
from typing import Dict, Sequence, Any
import pandas as pd
from .base_block import AnimalBlock
from .data_bundle import DataBundle, SAVE_DIR_BUNDLE_KEY
from ...pre_processing import SemanticScholar
from ...applications.Penguin import Penguin

from ...helpers.file_system import copy_all_files

class S2Block(AnimalBlock):
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
        tag: str = "S2",
        *,
        init_settings: Dict[str, Any] = None,
        call_settings: Dict[str, Any] = None,
        **kw,
    ) -> None:
        self.use_penguin = use_penguin
        self.penguin_settings = penguin_settings

        default_init = {
            'key': None,       # S2 API key
            'mode': 'fs',      # filesystem caching
            'name': None,      # real cache dir if provided
            'ignore': None,
            'verbose': True,
        }
        super().__init__(
            needs=needs,
            provides=provides,
            init_settings=self._merge(default_init, init_settings),
            call_settings={},
            tag=tag,
            **kw,
        )


    def run(self, bundle: DataBundle) -> None:
        # 1) Load input & extract DOIs
        raw = bundle[self.needs[0]]
        df_in = self.load_path(raw) if isinstance(raw, (str, Path)) else raw.copy()
        requested = df_in.doi.dropna().astype(str).tolist()
        if self.verbose:
            print(f"[{self.tag}] candidate DOIs count: {len(requested)}")

        # 2) Prepare output dir & CSV
        out_dir = Path(bundle[SAVE_DIR_BUNDLE_KEY]) / self.tag
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / "s2_df.csv"
        cache_dir = out_dir / "S2_CACHE"
        cache_dir.mkdir(exist_ok=True, parents=True)
        if self.verbose:
            print(f"[{self.tag}] cache dir: {cache_dir}")

        # 3) What’s already in Penguin?
        if self.use_penguin:
            penguin = Penguin(**self.penguin_settings)
            seen_db = set(penguin.distinct_dois(dois=requested))
        else:
            seen_db = set()
        if self.verbose:
            print(f"[{self.tag}] seen in Mongo: {len(seen_db)}")

        # 4) What’s already on disk?
        if csv_path.exists():
            df_csv = pd.read_csv(csv_path)
            seen_local = set(df_csv.doi.dropna().astype(str).tolist())
        else:
            df_csv = pd.DataFrame()
            seen_local = set()
        if self.verbose:
            print(f"[{self.tag}] seen locally: {len(seen_local)}")

        # 5) Figure out DOIs to fetch
        to_fetch = [d for d in requested if d not in seen_local and d not in seen_db]
        if self.verbose:
            print(f"[{self.tag}] to fetch count: {len(to_fetch)}")

        # 6) Fetch missing via SemanticScholar
        df_new = pd.DataFrame(columns=df_csv.columns)
        if to_fetch:
            settings = dict(self.init_settings, name=str(cache_dir))
            s2 = SemanticScholar(**settings)
            df_api, s2_targets = s2.search(to_fetch, mode="paper", n=0)
            if self.verbose:
                print(f"[{self.tag}] fetched via API: {len(df_api)}")

            # 6a) upsert into Penguin
            if self.use_penguin and not df_api.empty:
                penguin.add_many_documents(cache_dir, source="S2", overwrite=False)
                if self.verbose:
                    print(f"[{self.tag}] bulk-upserted JSON files into Mongo")

                # 6b) re-pull only those we just added
                # collect all S2 IDs from the API response
                df_api = penguin.id_search(
                    ids=[f"s2id:{uid}" for uid in set(s2_targets)],
                    as_pandas=True
                )
            df_new = df_api
            if self.verbose:
                print(f"[{self.tag}] entries from DB after upsert: {len(df_new)}")

        # 7) Pull any DB-only records we didn’t fetch locally
        df_db = pd.DataFrame(columns=df_csv.columns)
        if self.use_penguin:
            missing_db = seen_db - seen_local
            if missing_db:
                # search by DOI prefix
                df_id = penguin.id_search(
                    ids=[f"doi:{doi}" for doi in missing_db],
                    as_pandas=True
                )
                df_db = df_id[df_id.s2id.notnull()].reset_index(drop=True)
                if self.verbose:
                    print(f"[{self.tag}] added DB-only records: {len(df_db)}")

        # 8) Merge everything, dedupe on the S2 UID, save
        df_all = pd.concat([df_csv, df_new, df_db], ignore_index=True)
        df_all = df_all.drop_duplicates(subset=["s2id"]).reset_index(drop=True)
        df_all.to_csv(csv_path, index=False)
        if self.verbose:
            print(f"[{self.tag}] total after merge: {len(df_all)}")

        # 9) Move cache to real cache dir if requested
        real_cache = self.init_settings.get("name")
        if real_cache:
            copied = copy_all_files(cache_dir, real_cache, preserve_metadata=True)
            if self.verbose:
                print(f"[{self.tag}] copied {copied} files to real cache at {real_cache}")


        # 10) Checkpoint & attach
        self.register_checkpoint(self.provides[0], str(csv_path))
        bundle[f"{self.tag}.{self.provides[0]}"] = df_all
