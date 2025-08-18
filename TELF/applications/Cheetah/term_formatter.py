import os
import warnings
from typing import Dict, List, Tuple, Any, Optional, Set
from ...helpers.terms import resolve_substitution_conflicts


class CheetahTermFormatter:
    """
    Loads search terms from a Markdown file and returns them as
    plain strings or dict blocks, with optional category filtering.

    Optionally generates a substitution lookup map (with underscore variants),
    and can drop conflicts if requested.

    Parameters
    ----------
    markdown_file : str | Path
        Path to the .md file to load.
    lower : bool
        Whether to lowercase all term headers.
    category : str | None
        If set, include only `# Category: <category>` sections.
    include_general : bool
        If filtering by category, whether to include pre-category terms.
    substitutions : bool
        If True, builds substitution maps.
    all_categories : bool
        If True, overrides `category` and `include_general`.
    drop_conflicts : bool
        If True, resolve substitution conflicts and prune dropped entries.
        If False, keep all substitutions as-is (even if conflicting).
    """

    def __init__(
        self,
        markdown_file,
        lower: bool = False,
        category: Optional[str] = None,
        include_general: bool = True,
        substitutions: bool = False,
        all_categories: bool = False,
        drop_conflicts: bool = True,
    ):
        self.markdown_file    = markdown_file
        self.lower            = lower
        self.category         = category
        self.include_general  = include_general
        self.substitutions    = substitutions
        self.all_categories   = all_categories
        self.drop_conflicts   = drop_conflicts

        self.substitution_forward: Dict[str, str] = {}
        self.substitution_reverse: Dict[str, str] = {}

        # parse markdown → raw terms list
        self.terms: List[Any] = self._parse_markdown()

        # optionally build lookup tables
        if self.substitutions:
            self._build_substitutions_lookup()
            if self.drop_conflicts:
                self._postprocess_conflicts()

    # ──────────────────────────────────────────────────────────────── #
    # markdown parsing                                                #
    # ──────────────────────────────────────────────────────────────── #
    def _parse_markdown(self) -> List[Any]:
        terms: List[Any] = []
        current_term, positives, negatives = None, [], []
        active_block, current_section = False, None

        try:
            with open(self.markdown_file, "r", encoding="utf-8") as f:
                lines = f.readlines()
        except FileNotFoundError:
            warnings.warn(f"File '{self.markdown_file}' not found. Returning empty list.")
            return []

        for raw in lines:
            line = raw.strip()

            if line.startswith("# Category:"):
                current_section = line.split(":", 1)[1].strip()
                continue

            include_section = self.all_categories or self.category is None
            if self.category and not self.all_categories:
                include_section = (current_section == self.category) or (
                    current_section is None and self.include_general
                )

            if line.startswith("##"):
                if current_term is not None and active_block:
                    if positives or negatives:
                        terms.append({current_term: {"positives": positives, "negatives": negatives}})
                    else:
                        terms.append(current_term)

                positives, negatives = [], []
                header = line.lstrip("#").strip()
                if self.lower:
                    header = header.lower()
                current_term  = header
                active_block  = include_section

            elif active_block and line.lower().startswith("positives:"):
                items = [i.strip() for i in line.split(":", 1)[1].split(",") if i.strip()]
                positives.extend(items)

            elif active_block and line.lower().startswith("negatives:"):
                items = [i.strip() for i in line.split(":", 1)[1].split(",") if i.strip()]
                negatives.extend(items)

        if current_term is not None and active_block:
            if positives or negatives:
                terms.append({current_term: {"positives": positives, "negatives": negatives}})
            else:
                terms.append(current_term)

        return terms

    # ──────────────────────────────────────────────────────────────── #
    # substitutions lookup                                            #
    # ──────────────────────────────────────────────────────────────── #
    def _build_substitutions_lookup(self) -> None:
        """Create forward & reverse maps (no filtering yet)."""
        for entry in self.terms:
            if isinstance(entry, str):
                term = entry
                underscored = term.replace(" ", "_")
                self.substitution_forward[term] = underscored
                self.substitution_reverse[underscored] = term
            else:  # dict
                for term in entry.keys():
                    underscored = term.replace(" ", "_")
                    self.substitution_forward[term] = underscored
                    self.substitution_reverse[underscored] = term

    def _postprocess_conflicts(self) -> None:
        """Resolve substitution conflicts and prune dropped terms."""
        clean_forward, dropped = resolve_substitution_conflicts(
            self.substitution_forward, warn=True
        )
        self.substitution_forward = clean_forward

        # rebuild reverse map
        rev: Dict[str, List[str]] = {}
        for src, tgt in clean_forward.items():
            rev.setdefault(tgt, []).append(src)
        self.substitution_reverse = rev

        if not dropped:
            return

        # prune self.terms to match cleaned substitutions
        pruned_terms: List[Any] = []
        for entry in self.terms:
            if isinstance(entry, str):
                if entry not in dropped:
                    pruned_terms.append(entry)
            else:
                kept = {k: v for k, v in entry.items() if k not in dropped}
                if kept:
                    pruned_terms.append(kept)
        self.terms = pruned_terms

    # ──────────────────────────────────────────────────────────────── #
    # public access                                                   #
    # ──────────────────────────────────────────────────────────────── #
    def get_terms(self) -> List[Any]:
        return self.terms

    def get_substitution_maps(self) -> Tuple[Dict[str, str], Dict[str, str]]:
        return self.substitution_forward, self.substitution_reverse

   
    # ──────────────────────────────────────────────────────────────── #
    # public helpers                                                  #
    # ──────────────────────────────────────────────────────────────── #
    def get_terms(self):
        return self.terms

    def get_substitution_maps(self):
        """Return (forward_map, reverse_map)."""
        return self.substitution_forward, self.substitution_reverse


# ═══════════════════════════════════════════════════════════════════ #
# utility: convert TXT dump → cheetah markdown                       #
# ═══════════════════════════════════════════════════════════════════ #
def convert_txt_to_cheetah_markdown(txt_path, markdown_path):
    """
    Helper to convert a simple TXT list (optionally containing dict literals)
    into the markdown format expected by CheetahTermFormatter.
    """
    import ast

    with open(txt_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    md_lines: List[str] = []

    for line in lines:
        if line.startswith("{") and line.endswith("}"):
            try:
                parsed = ast.literal_eval(line)
                for key, value in parsed.items():
                    positives = [v.lstrip("+") for v in value if v.startswith("+")]
                    negatives = [v for v in value if not v.startswith("+")]
                    md_lines.append(f"## {key}")
                    if positives:
                        md_lines.append(f"positives: {', '.join(positives)}")
                    if negatives:
                        md_lines.append(f"negatives: {', '.join(negatives)}")
            except Exception as e:
                print(f"Skipping line due to parse error: {line}\nError: {e}")
        else:
            md_lines.append(f"## {line.strip()}")

    with open(markdown_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    print(f"Converted markdown saved to: {markdown_path}")
