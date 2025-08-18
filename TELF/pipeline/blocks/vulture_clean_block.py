# blocks/vulture_clean_block.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence, Any

from ...pre_processing.Vulture import Vulture
from ...pre_processing.Vulture.modules import (
    RemoveNonEnglishCleaner,
    SimpleCleaner,
    LemmatizeCleaner,
    SubstitutionCleaner,
)
from ...pre_processing.Vulture.default_stop_words import STOP_WORDS
from ...pre_processing.Vulture.default_stop_phrases import STOP_PHRASES

from .base_block import AnimalBlock
from .data_bundle import DataBundle, SAVE_DIR_BUNDLE_KEY

# from ...helpers.terms import resolve_substitution_conflicts
import spacy

class VultureCleanBlock(AnimalBlock):
    """
    Text-clean a DataFrame with **Vulture**.

    ─────────────────────────────────────────────────────────────
    always needs      : ('df',)     – takes the *latest* DataFrame
    needs  'substitutions' only when  use_substitutions=True
    provides   : ('df', 'vulture_steps')     – writes a cleaned copy
    tag        : 'VultureClean'   (namespace for its outputs)
    """
    CANONICAL_NEEDS = ("df",)

    def __init__(
        self,
        *,
        needs: Sequence[str] = CANONICAL_NEEDS,
        provides: Sequence[str] = ("df", "vulture_steps",),
        checkpoint_keys=("df",), 
        use_substitutions:bool = False,
        conditional_needs: Sequence[tuple[str, Any]] = (),   # none today
        tag: str = "VultureClean",
        init_settings: Dict[str, Any] | None = None,
        call_settings: Dict[str, Any] | None = None,
        **kw,
    ) -> None:

        default_init = {
            "n_jobs": -1,
            "verbose": 10,
            'parallel_backend':'threading'
        }
        self.use_substitutions = use_substitutions


        #nlp = spacy.load("en_core_web_sm")
        #for pipe in ["tok2vec","transformer","parser","ner","attribute_ruler","tagger"]:
        #    if pipe in nlp.pipe_names:
        #        nlp.remove_pipe(pipe)

        # now only ['lemmatizer'] (and the implicit tokenizer) remain
        lemmatize_cleaner = LemmatizeCleaner("spacy")
        #lemmatize_cleaner.backend = nlp

        self.steps = [
            RemoveNonEnglishCleaner(ascii_ratio=0.9, stopwords_ratio=0.25),
            SimpleCleaner(
                stop_words=STOP_WORDS,
                stop_phrases=STOP_PHRASES,
                order=[
                    "standardize_hyphens",
                    "isolate_frozen",
                    "remove_copyright_statement",
                    "remove_stop_phrases",
                    "make_lower_case",
                    "remove_formulas",
                    "normalize",
                    "remove_next_line",
                    "remove_email",
                    "remove_()",
                    "remove_[]",
                    "remove_special_characters",
                    "remove_nonASCII_boundary",
                    "remove_nonASCII",
                    "remove_tags",
                    "remove_stop_words",
                    "remove_standalone_numbers",
                    "remove_extra_whitespace",
                    "min_characters",
                    "remove_numbers",
                    "remove_alphanumeric",
                    "remove_roman_numerals",
                ],
            ),
            lemmatize_cleaner,
        ]
        default_call = {
            "columns": ["title", "abstract"],
            "append_to_original_df": True,
            "concat_cleaned_cols": True,
            "steps": self.steps
        }

        conds = list(conditional_needs or [])
        if self.use_substitutions:
            conds.append(("substitutions", lambda _b, _s: True))

        super().__init__(
            needs=needs,
            provides=provides,
            conditional_needs=conds,
            checkpoint_keys=checkpoint_keys,          # ← add this line
            tag=tag,
            init_settings=self._merge(default_init, init_settings),
            call_settings=self._merge(default_call, call_settings),
            **kw,
        )

    # ------------------------------------------------------------------ #
    # internal helper                                                    #
    # ------------------------------------------------------------------ #
    def __rebuild_steps(self, bundle: DataBundle):
        """Reconstruct the full steps list, including substitutions if enabled."""
        if self.use_substitutions:
            subs_map = self.load_path(bundle["substitutions"])
            initial = SubstitutionCleaner(subs_map, permute=True, lower=True, lemmatize=True)
            final   = SubstitutionCleaner(subs_map, permute=False, lower=False, lemmatize=True)
            return [initial] + self.steps + [final]
        else:
            return self.steps

    # ------------------------------------------------------------------ #
    # work                                                               #
    # ------------------------------------------------------------------ #
    def run(self, bundle: DataBundle) -> None:
        df = self.load_path(bundle[self.needs[0]])
        call_cfg = dict(self.call_settings)

        steps = self.__rebuild_steps(bundle)
        call_cfg["steps"] = steps

        vulture = Vulture(**self.init_settings)
        cleaned_df = vulture.clean_dataframe(df, **call_cfg)

        if SAVE_DIR_BUNDLE_KEY in bundle:
            out_dir = Path(bundle[SAVE_DIR_BUNDLE_KEY]) / self.tag
            out_dir.mkdir(exist_ok=True, parents=True)
            final_csv = out_dir / "clean_df.csv"
            cleaned_df.to_csv(final_csv, index=False, encoding="utf-8-sig")
            self.register_checkpoint(self.provides[0], final_csv)

        bundle[f"{self.tag}.{self.provides[0]}"] = cleaned_df
        bundle[f"{self.tag}.{self.provides[1]}"] = steps

    def _after_checkpoint_skip(self, bundle: DataBundle) -> None:
        """Re-insert fast-to-make steps after checkpoint reload."""
        steps = self.__rebuild_steps(bundle)
        bundle[f"{self.tag}.{self.provides[1]}"] = steps