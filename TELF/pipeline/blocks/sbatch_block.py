# blocks/sbatch_block.py
from __future__ import annotations
import subprocess, textwrap
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

import jsonpickle
from .base_block import AnimalBlock
from .data_bundle import DataBundle, SAVE_DIR_BUNDLE_KEY

# ───────────────────────── helpers ──────────────────────────

def _nospace(s: str) -> str:
    """Make a safe SLURM job-name / path fragment by replacing spaces."""
    return s.replace(" ", "_")

def clean_path(p: Path | str) -> Path:
    """Recursively apply _nospace to every component of the path."""
    p = Path(p)
    return Path(*(_nospace(part) for part in p.parts))

def shquote(x: str | Path) -> str:  # cheap, POSIX-only quoting
    return f'"{x}"'

# Default SLURM flags; the caller may override any of them via slurm_config
DEFAULT_SLURM: Dict[str, str] = {
    "partition": "production",
    "output":    "slurm-%j.out",
}


class SBatchBlock(AnimalBlock):
    """
    Wrap *any* AnimalBlock and run it via `sbatch`.

    Behaviour
    ---------
    • **Skip submission** if every declared checkpoint already exists.  
    • Otherwise:
      1. create a staging dir `<SAVE_DIR>/<tag>` (spaces → `_`);
      2. serialize the current bundle + wrapped block there;
      3. write a `run_block.py` runner     (faulthandler enabled);
      4. write a hardened `run.sbatch`     (threads pinned, Arrow off);
      5. `sbatch run.sbatch` and `sys.exit` so the pipeline resumes later.
    """
    def __init__(
        self,
        wrapped_block: AnimalBlock,
        *,
        venv_type: str = "conda",            # "conda" | "venv" | "poetry"
        venv_path: str = "TELF",             # env name or path
        slurm_config: Dict[str, str] | None = None,
        needs: Sequence[str] = (),
        provides: Sequence[str] = (),
        # tag: str = "SBatchWrapper",
        conditional_needs: Sequence[Tuple[str, Any]] = (),
        init_settings: Dict[str, Any] | None = None,
        call_settings: Dict[str, Any] | None = None,
        **kw,
    ) -> None:
        tag=wrapped_block.tag
        super().__init__(
            needs=needs or wrapped_block.needs,
            provides=provides or wrapped_block.provides,
            tag=tag,
            conditional_needs=conditional_needs,
            init_settings=init_settings or {},
            call_settings=call_settings or {},
            **kw,
        )
        self.wrapped_block = wrapped_block
        self.venv_type     = venv_type.lower()
        self.venv_path     = venv_path
        self.slurm         = {**DEFAULT_SLURM, **(slurm_config or {})}

    def run(self, bundle: DataBundle) -> None:
        # 1) fast-skip on existing checkpoints
        ck_keys = getattr(self.wrapped_block, "checkpoint_keys", ())
        if ck_keys and all(
            (f"{self.wrapped_block.tag}.{ck}") in bundle and
            Path(bundle[f"{self.wrapped_block.tag}.{ck}"]).exists()
            for ck in ck_keys
        ):
            print(f"⏭ {self.tag}: existing checkpoints – skipping submission.")
            try:
                self.wrapped_block._after_checkpoint_skip(bundle)
            except AttributeError:
                pass
            return

        # 2) staging directory
        base_dir = Path(bundle[SAVE_DIR_BUNDLE_KEY]).resolve()
        workdir  = clean_path(base_dir / self.tag)
        workdir.mkdir(parents=True, exist_ok=True)

        # adjust SLURM output path
        orig_out = self.slurm.get("output", DEFAULT_SLURM["output"])
        self.slurm["output"] = str(workdir / orig_out)

        # 2b) resume if runner already completed
        done = workdir / "_complete.json"
        if done.exists():
            # runner wrote out a small dict: { provide_name: value, ... }
            resumed: dict = jsonpickle.decode(done.read_text())
            wrapped_tag = self.wrapped_block.tag
            for p in self.provides:
                ns = f"{wrapped_tag}.{p}"
                if ns not in bundle:
                    bundle[ns] = resumed[p]
            return

        # ensure SAVE_DIR is absolute
        bundle[SAVE_DIR_BUNDLE_KEY] = str(base_dir)

        # 3) serialize bundle + block
        (workdir / "input_bundle.json").write_text(
            jsonpickle.encode(bundle), "utf-8"
        )
        (workdir / "block.json").write_text(
            jsonpickle.encode(self.wrapped_block), "utf-8"
        )

        # 4) runner script
        runner = workdir / "run_block.py"
        runner.write_text(textwrap.dedent(f"""
            import faulthandler, jsonpickle
            faulthandler.enable()

            bundle = jsonpickle.decode(open(r"{workdir/'input_bundle.json'}").read())
            block  = jsonpickle.decode(open(r"{workdir/'block.json'}").read())
            out_bundle = block(bundle)

            # collect just the wrapped block's provides
            result = {{}}
            for p in {list(self.provides)!r}:
                result[p] = out_bundle[f"{{block.tag}}.{{p}}"]

            open(r"{done}", "w").write(jsonpickle.encode(result))
        """).strip(), "utf-8")

        # 5) sbatch script
        sbatch = workdir / "run.sbatch"
        lines  = [
            "#!/bin/bash",
            f"#SBATCH --job-name={_nospace(self.tag)}",
            *(f"#SBATCH --{k}={v}" for k, v in self.slurm.items()),
            "",
            "ulimit -c unlimited",
            "export OMP_NUM_THREADS=1",
            "export OPENBLAS_NUM_THREADS=1",
            "export MKL_NUM_THREADS=1",
            "export NUMEXPR_NUM_THREADS=1",
            "export MKL_THREADING_LAYER=GNU",
            "export OPENBLAS_DISABLE_THREADS=1",
            "export PANDAS_ARROW_DISABLED=1",
            "",
            f"cd {shquote(workdir)}",
        ]

        match self.venv_type:
            case "venv":
                lines.append(f"source {shquote(Path(self.venv_path).resolve()/ 'bin'/'activate')}")
            case "conda":
                lines += [
                    "source $(conda info --base)/etc/profile.d/conda.sh",
                    f"conda activate {self.venv_path}",
                ]
            case "poetry":
                lines += [
                    f"cd {shquote(Path.cwd())}",
                    "poetry install --no-root --quiet",
                ]
            case _:
                raise ValueError(f"Unknown venv_type {self.venv_type!r}")

        lines.append(f"python -Xfaulthandler {shquote(runner)}")
        sbatch.write_text("\n".join(lines), "utf-8")

        # 6) submit and exit
        print(f"🚀 {self.tag}: submitting via sbatch …")
        subprocess.run(["sbatch", str(sbatch)], check=True)
        raise SystemExit(f"{self.tag} submitted – pipeline will resume after job finishes.")
