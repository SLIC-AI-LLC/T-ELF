from pathlib import Path
from typing import Any, Dict

from .base_block import AnimalBlock
from .data_bundle import DataBundle, SAVE_DIR_BUNDLE_KEY
from ...helpers.figures import create_wordcloud_from_df

class WordCloudBlock(AnimalBlock):
    CANONICAL_NEEDS = ('df', )
    
    def __init__(
        self,
        needs=CANONICAL_NEEDS,
        provides=("wordcloud",),
        tag='Wordcloud',
        *,
        init_settings: Dict[str, Any] = None,
        call_settings: Dict[str, Any] = None,
        **kw
    ):
        default_init = {}
        default_call = {
            'col': 'clean_title_abstract',
            'n': 30,
            'save_path': None,
            'figsize': (6, 6),
        }

        super().__init__(
            needs=needs,
            provides=provides,
            tag=tag,
            init_settings=self._merge(default_init, init_settings),
            call_settings=self._merge(default_call, call_settings),
            **kw,
        )

    def run(self, bundle: DataBundle) -> None:
        # 1) Load the DataFrame and drop any rows where the target column is NaN
        df = (
            self.load_path(bundle[self.needs[0]])
            .dropna(subset=[self.call_settings['col']])
        )

        # 2) Generate the wordcloud (first positional arg is df)
        create_wordcloud_from_df(df, **self.call_settings)

        # 3) Decide on the output PNG path
        save_path = self.call_settings['save_path']
        if save_path is not None:
            out_path = Path(save_path)
        else:
            out_path = Path(bundle[SAVE_DIR_BUNDLE_KEY]) / self.tag

        # 4) “Load” that path (or simply return it) and record in the bundle
        png = self.load_path(out_path)
        bundle[f"{self.tag}.{self.provides[0]}"] = png
