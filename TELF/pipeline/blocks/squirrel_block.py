from pathlib import Path
from typing import Dict, Sequence, Any, Tuple
import pandas as pd

from .base_block import AnimalBlock
from .data_bundle import DataBundle, SAVE_DIR_BUNDLE_KEY

from ...pre_processing import Squirrel
from ...pre_processing.Squirrel.pruners import EmbeddingPruner
from ...pre_processing.Squirrel.pruners import LLMPruner


class SquirrelBlock(AnimalBlock):
    CANONICAL_NEEDS = ('df', )
    
    def __init__(
        self,
        needs = CANONICAL_NEEDS,
        provides = ("df",),
        tag: str = "Squirrel",
        low_count_backup: int = 25,
        conditional_needs: Sequence[Tuple[str, Any]]  = (),
        *,
        init_settings: Dict[str, Any] = None,
        call_settings: Dict[str, Any] = None,
        **kw,
    ) -> None:
        self.low_count_backup = low_count_backup
        
        emb_pruner = EmbeddingPruner(
            embedding_model="SPECTER",
            distance_std_factor=5.0,
            overwrite_embeddings=False,
            use_gpu=True,
            verbose=True,
        )
        llm_pruner = LLMPruner(
            llm_model_name="llama3.2:latest",
            llm_api_url="http://localhost:11434",
            llm_vote_trials=4,
            llm_promote_threshold=0.75,
            llm_temperature=0.7,
            verbose=True,
        )
        self.pipeline = [emb_pruner, llm_pruner]
        self.backup_pipeline = [llm_pruner]
        default_init = {
            'data_column':  'title_abstract',
            'label_column': 'type',
            'reference_label': 0,
            'aggregrate_prune': True,
            'pipeline':self.pipeline
        }
        default_call = {}

        super().__init__(
            needs=needs,
            provides=provides,
            tag=tag,
            init_settings=self._merge(default_init, init_settings),
            call_settings=self._merge(default_call, call_settings),
            conditional_needs=conditional_needs,
            **kw,
        )
        
    def run(self, bundle: DataBundle) -> None:
        raw = bundle[self.needs[0]]
        if isinstance(raw, (str, Path)):
            df = self.load_path(raw)
        else:
            df = raw.copy()
        
        # If there are less than 25 documents, use only llm pipeline
        label_col   = self.init_settings['label_column']
        ref_label   = self.init_settings['reference_label']
        count_ref   = int((df[label_col] == ref_label).sum())
        if self.low_count_backup and count_ref < self.low_count_backup:
            self.init_settings['pipeline'] = self.backup_pipeline

        if self.init_settings['data_column'] not in df.columns:
            try:
                # df[self.init_settings['data_column']] = df['title'] + ' ' + df['abstract']
                df[self.init_settings['data_column']] = df['title'].astype(str) + ' ' + df['abstract'].astype(str)
            except KeyError:
                raise ValueError(f"Data column {self.init_settings['data_column']} not found in DataFrame.")
            except Exception as e:
                raise RuntimeError(f"An error occurred while creating the data column: {e}")

        squirrel_dir = Path(bundle[SAVE_DIR_BUNDLE_KEY]) / self.tag
        squirrel = Squirrel(data_source=df, 
                            output_dir=squirrel_dir, 
                            **self.init_settings)
        squirrel()
        expected_csv = squirrel_dir / 'squirrel_pruned.csv'

        result_df = pd.read_csv(expected_csv)
        self.register_checkpoint(self.provides[0], expected_csv)

        bundle[f"{self.tag}.{self.provides[0]}"] = result_df
