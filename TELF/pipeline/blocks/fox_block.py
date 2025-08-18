from pathlib import Path
from typing import Dict, Sequence, Any
import os
from .base_block import AnimalBlock
from .data_bundle import DataBundle, SAVE_DIR_BUNDLE_KEY
from ...post_processing import Fox

class FoxBlock(AnimalBlock):
    """
    Post-processing

    ─────────────────────────────────────────────────────────────
    always needs      : ("df", "vocabulary", "model_path", "k", ),
    provides   : ('df',)
    tag        : 'Fox' 
    """
    CANONICAL_NEEDS = ("df", "vocabulary", "model_path", "k", ),

    def __init__(
        self,
        API_KEY:str=None,
        needs: Sequence[str] = CANONICAL_NEEDS,
        provides: Sequence[str] = ("df", ),
        make_summaries:bool=True,
        conditional_needs: Sequence[tuple[str, Any]] = (), 
        tag: str = "Fox",
        init_settings: Dict[str, Any] | None = None,
        call_settings: Dict[str, Any] | None = None,
        **kw,
    ):  

        self.API_KEY=API_KEY
        self.tag=tag
        self.make_summaries = make_summaries
        default_init = {
            "summary_model":"gpt-3.5-turbo-instruct",
            "verbose":False,
            "debug":True,
        }
        default_call = {}
        super().__init__(
            needs=needs,
            provides=provides,
            conditional_needs=conditional_needs,
            tag=tag,
            init_settings=self._merge(default_init, init_settings),
            call_settings=self._merge(default_call, call_settings),
            **kw,
        )


    def run(self, bundle: DataBundle) -> None:
        df = self.load_path(bundle[self.needs[0]])
        vocabulary = self.load_path(bundle[self.needs[1]])
        model_path = self.load_path(bundle[self.needs[2]])
        k = self.load_path(bundle[self.needs[3]])

        if SAVE_DIR_BUNDLE_KEY in bundle:
            out_dir = Path(bundle[SAVE_DIR_BUNDLE_KEY]) / self.tag
            out_dir.mkdir(exist_ok=True, parents=True)
        else:
            out_dir = self.tag            

        fox = Fox(**self.init_settings)

        post_processed_df_path = fox.post_process(
            npz_path=os.path.join(model_path, f"WH_k={k}.npz"),
            vocabulary=vocabulary,
            data=df,                
            output_dir=out_dir
        )

        if self.make_summaries:
            fox.setApiKey(self.API_KEY)
            fox.makeSummariesAndLabelsOpenAi(
                processing_path=post_processed_df_path 
            )

        bundle[f"{self.tag}.{self.provides[0]}"] = post_processed_df_path
