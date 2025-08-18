# blocks/function_block.py
from __future__ import annotations

from typing import Any, Callable, Dict, Tuple, Sequence

from .data_bundle import DataBundle
from .base_block import AnimalBlock


class FunctionBlock(AnimalBlock):
    """
    Wrap **any** Python callable so it becomes a pipeline block.

    The callable is invoked as

        result = function_call(*[bundle[n] for n in needs], **call_settings)

    and its return value(s) are written back to the bundle under this block’s
    namespace (`tag.provides[i]`).

    Because the class derives from *AnimalBlock* it automatically supports:

    * **tagging** – no two blocks clobber each other’s outputs.
    * **conditional_needs** – declare runtime-dependent inputs if you ever need
      them (none by default).
    """

    # ------------------------------------------------------------------ #
    # constructor                                                        #
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        *,
        needs: Tuple[str, ...],
        provides: Tuple[str, ...],
        function_call: Callable[..., Any],
        call_settings: Dict[str, Any] | None = None,
        tag: str = "Function",
        conditional_needs: Sequence[Tuple[str, Any]] = (),  # optional
        **kw,
    ) -> None:
        super().__init__(
            needs=needs,
            provides=provides,
            conditional_needs=conditional_needs,
            tag=tag,
            call_settings=call_settings or {},
            **kw,
        )

        self.function_call = function_call

    # ------------------------------------------------------------------ #
    # work                                                                #
    # ------------------------------------------------------------------ #
    def run(self, bundle: DataBundle) -> None:
        # build positional argument list from declared needs
        args = [bundle[key] for key in self.needs]

        # kwargs to pass through
        kwargs = (
            self.call_settings
            if isinstance(self.call_settings, dict)
            else vars(self.call_settings)
        )

        # call the user function
        result = self.function_call(*args, **kwargs)

        # write back under this block’s namespace
        if len(self.provides) == 1:
            bundle[f"{self.tag}.{self.provides[0]}"] = result
        else:
            if not isinstance(result, (list, tuple)) or len(result) != len(self.provides):
                raise ValueError(
                    f"{self.__class__.__name__} expected {len(self.provides)} return "
                    f"values but got {type(result)} with length {getattr(result, '__len__', lambda: '?')()}"
                )
            for key, val in zip(self.provides, result):
                bundle[f"{self.tag}.{key}"] = val
