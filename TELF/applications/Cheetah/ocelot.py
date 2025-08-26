#!/usr/bin/env python3
"""
ngram_filter_class.py

Fast phrase filter with per-main constraints + optional global constraints.

BLOCK RULES (default):
  Each '## <main>' defines a rule. The 'positives:' / 'negatives:' lines that follow
  apply only to that main. A row passes if ANY rule passes AND it also satisfies the
  GLOBAL constraints (if present).

GLOBAL CONSTRAINTS:
  A heading named 'Global' (any # level; case-insensitive; aliases supported) may contain
  'positives:' and/or 'negatives:' lines. These are required for EVERY rule.

MODES:
  - Rule-level positives_mode:   "any" (default) or "all"
  - Global positives mode:       "any" (default) or "all"

Regex matching:
  - Phrase tokens are joined with \W+ so they cross spaces/punct/newlines
  - Word-ish boundaries avoid substrings ('apple' won't match 'pineapple')
  - Case-insensitive
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Pattern, Sequence, Tuple, Optional, Set


# ---------- Data structures ----------

@dataclass
class MatchDetail:
    passed: bool
    matched_main: List[str]
    matched_positives: List[str]
    matched_negatives: List[str]

@dataclass
class Rule:
    main_phrases: List[str]
    positive_phrases: List[str]
    negative_phrases: List[str]
    positives_mode: str = "any"  # "any" or "all"


# ---------- Class ----------

class Ocelot:
    """
    Encapsulates parsing, pattern compilation, matching, and CSV processing.
    - Per-main positives/negatives (block rules)
    - Optional GLOBAL positives/negatives required for every rule
    """

    SECTION_ALIASES = {
        "main": {"main", "main search", "main search terms", "mains", "core"},
        "positives": {"positives", "positive", "pos", "include", "includes"},
        "negatives": {"negatives", "negative", "neg", "exclude", "excludes", "blacklist"},
    }
    GLOBAL_ALIASES = {"global", "globals", "global constraints", "global rules"}

    def __init__(
        self,
        rules: Sequence[Rule] | None = None,
        *,
        global_positives: Sequence[str] | None = None,
        global_negatives: Sequence[str] | None = None,
        global_positives_mode: str = "any",   # "any" or "all"
    ) -> None:
        self.rules: List[Rule] = list(rules or [])

        # Global constraints
        if global_positives_mode not in {"any", "all"}:
            raise ValueError("global_positives_mode must be 'any' or 'all'")
        self.global_positives_mode: str = global_positives_mode
        self.global_positives: List[str] = list(global_positives or [])
        self.global_negatives: List[str] = list(global_negatives or [])

        # Compiled per-rule + compiled global
        self._compiled: List[Tuple[Optional[Pattern], Dict[str, int],
                                   Optional[Pattern], Dict[str, int],
                                   Optional[Pattern], Dict[str, int]]] = []
        self._any_main_rx: Optional[Pattern] = None

        self._gpos_rx: Optional[Pattern] = None
        self._gpos_gmap: Dict[str, int] = {}
        self._gneg_rx: Optional[Pattern] = None
        self._gneg_gmap: Dict[str, int] = {}

        self.compile_patterns()

    # ---------- Construction helpers ----------

    @classmethod
    def from_markdown(
        cls,
        md_text: str,
        *,
        positives_mode: str = "any",
        global_positives_mode: str = "any",
    ) -> "Ocelot":
        rules, gpos, gneg = cls.parse_rules_markdown(
            md_text,
            default_positives_mode=positives_mode,
        )
        return cls(
            rules,
            global_positives=gpos,
            global_negatives=gneg,
            global_positives_mode=global_positives_mode,
        )

    @classmethod
    def from_markdown_file(
        cls,
        path: str | Path,
        *,
        positives_mode: str = "any",
        global_positives_mode: str = "any",
    ) -> "Ocelot":
        md_text = Path(path).read_text(encoding="utf-8")
        return cls.from_markdown(
            md_text,
            positives_mode=positives_mode,
            global_positives_mode=global_positives_mode,
        )

    # ---------- Parsing (BLOCK rules + optional GLOBAL section) ----------

    @staticmethod
    def _normalize_heading(text: str) -> str:
        return re.sub(r"\s+", " ", text.strip().lower())

    @classmethod
    def parse_rules_markdown(
        cls,
        md_text: str,
        default_positives_mode: str = "any",
    ) -> Tuple[List[Rule], List[str], List[str]]:
        """
        BLOCK mode with an optional distinct GLOBAL section.

        - Every heading becomes either:
          * a new Rule (per-main), or
          * the GLOBAL section if its normalized text is in GLOBAL_ALIASES.

        - Inside a Rule or GLOBAL section, collect:
          positives: a, b  (also supports bullets under the label)
          negatives: c, d

        Returns (rules, global_positives, global_negatives)
        """
        def _dedupe(seq: List[str]) -> List[str]:
            seen = set(); out = []
            for s in seq:
                k = s.casefold()
                if k not in seen:
                    seen.add(k); out.append(s)
            return out

        re_heading = re.compile(r"^\s{0,3}#{1,6}\s+(.*\S)\s*$")
        re_list    = re.compile(r"^\s*[-*+]\s+(.*\S)\s*$")
        re_label   = re.compile(r"^\s*(positives?|negatives?)\s*:\s*(.*)\s*$", re.IGNORECASE)

        rules: List[Rule] = []
        current_rule: Optional[Rule] = None
        current_block_key: Optional[str] = None  # 'positives' | 'negatives'
        in_global: bool = False
        global_pos: List[str] = []
        global_neg: List[str] = []

        def _flush_rule():
            nonlocal current_rule
            if current_rule:
                current_rule.main_phrases     = _dedupe(current_rule.main_phrases)
                current_rule.positive_phrases = _dedupe(current_rule.positive_phrases)
                current_rule.negative_phrases = _dedupe(current_rule.negative_phrases)
                rules.append(current_rule)
                current_rule = None

        for raw in md_text.splitlines():
            line = raw.rstrip()

            m_h = re_heading.match(line)
            if m_h:
                # new section starts
                _flush_rule()
                current_block_key = None
                head = m_h.group(1).strip()
                norm = cls._normalize_heading(head)

                if norm in cls.GLOBAL_ALIASES:
                    in_global = True
                    continue

                # regular rule heading (per-main)
                in_global = False
                current_rule = Rule(
                    main_phrases=[head],
                    positive_phrases=[],
                    negative_phrases=[],
                    positives_mode=default_positives_mode,
                )
                continue

            # Inside GLOBAL
            if in_global:
                m_lab = re_label.match(line)
                if m_lab:
                    label = m_lab.group(1).lower()
                    rest = m_lab.group(2).strip()
                    current_block_key = "positives" if label.startswith("pos") else "negatives"
                    if rest:
                        items = [x.strip() for x in rest.split(",")]
                        if current_block_key == "positives":
                            global_pos.extend([i for i in items if i])
                        else:
                            global_neg.extend([i for i in items if i])
                    continue

                m_li = re_list.match(line)
                if m_li and current_block_key:
                    item = m_li.group(1).strip()
                    if current_block_key == "positives":
                        global_pos.append(item)
                    else:
                        global_neg.append(item)
                    continue

                if line.strip() and current_block_key:
                    if current_block_key == "positives":
                        global_pos.append(line.strip())
                    else:
                        global_neg.append(line.strip())
                    continue

                # text before labels in global -> ignore
                continue

            # Inside a RULE
            if current_rule:
                m_lab = re_label.match(line)
                if m_lab:
                    label = m_lab.group(1).lower()
                    rest = m_lab.group(2).strip()
                    current_block_key = "positives" if label.startswith("pos") else "negatives"
                    if rest:
                        items = [x.strip() for x in rest.split(",")]
                        if current_block_key == "positives":
                            current_rule.positive_phrases.extend([i for i in items if i])
                        else:
                            current_rule.negative_phrases.extend([i for i in items if i])
                    continue

                m_li = re_list.match(line)
                if m_li and current_block_key:
                    item = m_li.group(1).strip()
                    if current_block_key == "positives":
                        current_rule.positive_phrases.append(item)
                    else:
                        current_rule.negative_phrases.append(item)
                    continue

                if line.strip() and current_block_key:
                    if current_block_key == "positives":
                        current_rule.positive_phrases.append(line.strip())
                    else:
                        current_rule.negative_phrases.append(line.strip())
                    continue

                # other text in rule before labels -> ignore
                continue

            # outside any recognized section -> ignore

        _flush_rule()
        # dedupe globals
        global_pos = _dedupe(global_pos)
        global_neg = _dedupe(global_neg)
        return rules, global_pos, global_neg

    # ---------- Pattern compilation ----------

    @staticmethod
    def phrase_to_pattern_str(phrase: str) -> str:
        phrase = phrase.strip()
        if not phrase:
            return ""
        parts = [re.escape(tok) for tok in phrase.split()]
        core = r"\W+".join(parts)
        return rf"(?<!\w)(?:{core})(?!\w)"

    @classmethod
    def _compile_group_pattern(cls, phrases: Sequence[str]) -> Tuple[Pattern | None, Dict[str, int]]:
        chunks: List[str] = []
        gmap: Dict[str, int] = {}
        for i, ph in enumerate(phrases):
            pat = cls.phrase_to_pattern_str(ph)
            if pat:
                g = f"g{i}"
                chunks.append(f"(?P<{g}>{pat})")
                gmap[g] = i
        if not chunks:
            return None, {}
        return re.compile("|".join(chunks), flags=re.IGNORECASE), gmap

    def compile_patterns(self) -> None:
        """Compile per-rule grouped regexes, any-main prefilter, and global patterns."""
        self._compiled = []
        all_mains: List[str] = []
        for rule in self.rules:
            main_rx, main_g = self._compile_group_pattern(rule.main_phrases)
            pos_rx,  pos_g  = self._compile_group_pattern(rule.positive_phrases)
            neg_rx,  neg_g  = self._compile_group_pattern(rule.negative_phrases)
            self._compiled.append((main_rx, main_g, pos_rx, pos_g, neg_rx, neg_g))
            all_mains.extend(rule.main_phrases)
        self._any_main_rx, _ = self._compile_group_pattern(all_mains)

        # Global patterns
        self._gpos_rx, self._gpos_gmap = self._compile_group_pattern(self.global_positives)
        self._gneg_rx, self._gneg_gmap = self._compile_group_pattern(self.global_negatives)

    # ---------- Matching ----------

    @staticmethod
    def _scan(rx: Optional[Pattern], gmap: Dict[str, int], text: str) -> Set[int]:
        if not rx:
            return set()
        out: Set[int] = set()
        for m in rx.finditer(text):
            out.add(gmap[m.lastgroup])  # type: ignore[index]
        return out

    def _global_ok(self, text: str) -> Tuple[bool, List[str], List[str]]:
        """Return (ok, matched_global_pos, matched_global_neg)."""
        matched_pos = [self.global_positives[i] for i in sorted(self._scan(self._gpos_rx, self._gpos_gmap, text))]
        matched_neg = [self.global_negatives[i] for i in sorted(self._scan(self._gneg_rx, self._gneg_gmap, text))]
        if matched_neg:
            return False, matched_pos, matched_neg
        if self.global_positives:
            if self.global_positives_mode == "all":
                ok = len(matched_pos) == len(self.global_positives)
            else:
                ok = bool(matched_pos)
            if not ok:
                return False, matched_pos, matched_neg
        return True, matched_pos, matched_neg

    def evaluate_text(self, text: str) -> MatchDetail:
        """
        Pass if ANY per-main rule passes AND global constraints are satisfied.
        Report the first passing rule's matches + any matched global negatives (if they caused failure).
        """
        if self._any_main_rx and not self._any_main_rx.search(text):
            return MatchDetail(False, [], [], [])

        # Check globals first; if they fail, we still may want to report which globals blocked.
        g_ok, g_pos_hits, g_neg_hits = self._global_ok(text)
        if not g_ok:
            # Fail fast; show global negatives (or lack of required positives) but no mains triggered
            return MatchDetail(False, [], g_pos_hits, g_neg_hits)

        for rule, (main_rx, main_g, pos_rx, pos_g, neg_rx, neg_g) in zip(self.rules, self._compiled):
            main_idxs = self._scan(main_rx, main_g, text)
            if not main_idxs:
                continue
            neg_idxs = self._scan(neg_rx, neg_g, text)
            if neg_idxs:
                continue

            if rule.positive_phrases:
                pos_idxs = self._scan(pos_rx, pos_g, text)
                if rule.positives_mode == "all":
                    ok_pos = len(pos_idxs) == len(rule.positive_phrases)
                else:
                    ok_pos = bool(pos_idxs)
                if not ok_pos:
                    continue
            else:
                pos_idxs = set()

            matched_main = [rule.main_phrases[i] for i in sorted(main_idxs)]
            matched_pos  = [rule.positive_phrases[i] for i in sorted(pos_idxs)]
            # Global positives are required but not reported as "matched_negatives"
            return MatchDetail(True, matched_main, matched_pos, [])

        return MatchDetail(False, [], [], [])

    # ---------- Plain text / JSONL iter ----------

    @staticmethod
    def iter_inputs(path: str | None, *, jsonl: bool, field: Optional[str]) -> Iterable[Tuple[int, str]]:
        stream = sys.stdin if path in (None, "-", "") else open(path, "r", encoding="utf-8", errors="replace")
        with stream:
            for idx, line in enumerate(stream):
                line = line.rstrip("\n")
                if jsonl:
                    try:
                        obj = json.loads(line)
                        if field is None:
                            raise ValueError("--field is required with --jsonl")
                        val = obj.get(field, "")
                        if not isinstance(val, str):
                            val = "" if val is None else str(val)
                        yield idx, val
                    except json.JSONDecodeError:
                        continue
                else:
                    yield idx, line

    # ---------- Text/JSONL to CSV (legacy) ----------

    def filter_to_csv(
        self,
        input_path: str | None,
        output_path: str | None,
        *,
        jsonl: bool = False,
        field: Optional[str] = None,
        emit_nonmatches: bool = False,
    ) -> None:
        out_stream = sys.stdout if output_path in (None, "-", "") else open(output_path, "w", newline="", encoding="utf-8")
        writer = csv.writer(out_stream)
        writer.writerow(["index", "passed", "matched_main", "matched_positives", "matched_negatives", "text"])

        for idx, text in self.iter_inputs(input_path, jsonl=jsonl, field=field):
            detail = self.evaluate_text(text)
            if detail.passed or emit_nonmatches:
                writer.writerow([
                    idx,
                    str(detail.passed).lower(),
                    "|".join(detail.matched_main),
                    "|".join(detail.matched_positives),
                    "|".join(detail.matched_negatives),
                    text,
                ])

        if out_stream is not sys.stdout:
            out_stream.close()

    # ---------- CSV processing (inside Ocelot) ----------

    def filter_csv_rows(
        self,
        rows: List[Dict[str, str]],
        *,
        text_field: str = "text",
        id_field: Optional[str] = None,
        emit_nonmatches: bool = False,
    ) -> List[Dict[str, str]]:
        out: List[Dict[str, str]] = []
        for idx, row in enumerate(rows):
            txt_raw = row.get(text_field, "")
            text = txt_raw if isinstance(txt_raw, str) else ("" if txt_raw is None else str(txt_raw))
            detail = self.evaluate_text(text)
            if detail.passed or emit_nonmatches:
                index_val = row.get(id_field) if id_field else idx
                out.append({
                    "index": index_val,
                    "passed": str(detail.passed).lower(),
                    "matched_main": "|".join(detail.matched_main),
                    "matched_positives": "|".join(detail.matched_positives),
                    "matched_negatives": "|".join(detail.matched_negatives),
                    "text": text,
                })
        return out

    def filter_csv_file(
        self,
        input_csv_path: str,
        output_csv_path: Optional[str] = None,
        *,
        text_field: str = "text",
        id_field: Optional[str] = None,
        emit_nonmatches: bool = False,
        encoding: str = "utf-8",
    ) -> List[Dict[str, str]]:
        with open(input_csv_path, "r", encoding=encoding, newline="") as f:
            reader = csv.DictReader(f)
            rows = [dict(r) for r in reader]

        results = self.filter_csv_rows(
            rows,
            text_field=text_field,
            id_field=id_field,
            emit_nonmatches=emit_nonmatches,
        )

        if output_csv_path:
            with open(output_csv_path, "w", encoding=encoding, newline="") as out:
                writer = csv.writer(out)
                writer.writerow(["index", "passed", "matched_main", "matched_positives", "matched_negatives", "text"])
                for r in results:
                    writer.writerow([r["index"], r["passed"], r["matched_main"], r["matched_positives"], r["matched_negatives"], r["text"]])

        return results
 