from pathlib import Path
from typing import Any, Dict, Sequence, Optional, Callable, Tuple

from .base_block import AnimalBlock
from .data_bundle import DataBundle, SAVE_DIR_BUNDLE_KEY

from TELF.pre_processing.Vulture.modules.substitute import SubstitutionCleaner
from TELF.pipeline.blocks import VultureCleanBlock
from TELF.helpers.terms import resolve_substitution_conflicts
from TELF.applications.Bunny import AutoBunny, AutoBunnyStep


class AutoBunnyBlock(AnimalBlock):
    CANONICAL_NEEDS = ('df','terms',)

    def __init__(
        self,
        num_hops: int = 5,
        cheetah_settings: Dict = {
            'in_title': False,
            'in_abstract': True,
            'and_search': False,
        },
        modes: Sequence[str]=['citations'],
        hop_priority: str= "frequency",
        needs: Tuple[str, str] = CANONICAL_NEEDS,
        provides: Tuple[str, ...] = ('df',),
        use_substitutions:bool=False,
        use_vulture_steps:bool=False,
        conditional_needs: Sequence[Tuple[str, Any]] = (),
        tag: str = "AutoBunny",
        *,
        init_settings: Optional[Dict[str, Any]] = None,
        call_settings: Optional[Dict[str, Any]] = None,
        verbose: bool = True,
        **kw: Any,
    ) -> None:
        """
        Dataset expansion tool

        ─────────────────────────────────────────────────────────────
        always needs      : ('df', 'terms')
        needs  'substitutions' only when  use_substitutions=True and use_vulture_steps=False
        needs  'vulture_steps' only when  use_vulture_steps=True
        provides   : ('df',) 
        tag        : 'AutoBunny'   (namespace for its outputs)
        """
        
        # Default init settings for AutoBunny
        default_init: Dict[str, Any] = {
            's2_key': None,
            'scopus_keys': None,
            'cache_dir': None,
            'verbose': True,
        }
        self.use_substitutions = use_substitutions
        self.use_vulture_steps = use_vulture_steps

        conds = list(conditional_needs or [])
        if self.use_substitutions and not self.use_vulture_steps:
            conds.append(("substitutions", lambda _b, _s: True))
        
        if self.use_vulture_steps:
            conds.append(("vulture_steps", lambda _b, _s: True))
            self.VULTURE_STEPS = None
        else:
            # Load the standard Vulture steps as a list
            self.VULTURE_STEPS = list(VultureCleanBlock().steps)

        # Placeholder for Cheetah settings (unused here)
        self.cheetah_settings = cheetah_settings

        # Build initial AutoBunnyStep list
        steps = [
            AutoBunnyStep(
                modes=modes,
                max_papers=0,
                hop_priority=hop_priority,
                vulture_settings=self.VULTURE_STEPS,
            )
            for _ in range(num_hops)
        ]

        default_call: Dict[str, Any] = {
            'steps': steps,
            'filter_type': 'AFFILCOUNTRY',
            'filter_value': None,
        }

        super().__init__(
            needs=needs,
            provides=provides,
            conditional_needs=conds,
            tag=tag,
            init_settings=self._merge(default_init, init_settings),
            call_settings=self._merge(default_call, call_settings),
            verbose=verbose,
            **kw,
        )

    def run(self, bundle: DataBundle) -> None:
        df = self.load_path(bundle[self.needs[0]])
        terms = self.load_path(bundle[self.needs[1]])
        self.cheetah_settings['query'] = terms

        if self.use_vulture_steps:
            self.VULTURE_STEPS = self.load_path(bundle["vulture_steps"])

            for step in self.call_settings['steps']:
                step.vulture_settings = self.VULTURE_STEPS
                step.cheetah_settings = self.cheetah_settings

        elif self.use_substitutions and not self.use_vulture_steps:
            substitutions = self.load_path(bundle["vulture_steps"])
            initial_sub = SubstitutionCleaner(substitutions, permute=True,  lower=True,  lemmatize=True)
            final_sub   = SubstitutionCleaner(substitutions, permute=False, lower=False, lemmatize=True)
            vchain = [initial_sub] + self.VULTURE_STEPS + [final_sub]

            for step in self.call_settings['steps']:
                step.vulture_settings = vchain
                step.cheetah_settings = self.cheetah_settings
            
        
        self.output_dir = Path(bundle[SAVE_DIR_BUNDLE_KEY]) / self.tag
        self.output_dir.mkdir(parents=True, exist_ok=True)

        ab = AutoBunny(core=df, output_dir=self.output_dir, **self.init_settings)
        ab_df = ab.run(**self.call_settings)

        bundle[f"{self.tag}.{self.provides[0]}"] = ab_df
